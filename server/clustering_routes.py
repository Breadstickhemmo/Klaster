import os
import uuid
import json
import logging
import time
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np
from werkzeug.utils import secure_filename
from models import db, ClusteringSession, ClusterMetadata
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import flag_modified
from clustering_logic import (
    run_clustering_pipeline,
    redistribute_cluster_data,
    log_manual_adjustment,
    load_embeddings,
    get_cluster_labels_for_session,
    generate_and_save_scatter_data
)

logger = logging.getLogger(__name__)

clustering_bp = Blueprint('clustering_api', __name__, url_prefix='/api/clustering')

ALLOWED_EXTENSIONS = {'parquet'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@clustering_bp.route('/start', methods=['POST'])
@jwt_required()
def start_clustering():
    current_user_id = get_jwt_identity()
    if 'embeddingFile' not in request.files:
        return jsonify({"error": "Файл эмбеддингов ('embeddingFile') не найден"}), 400
    file = request.files['embeddingFile']
    algorithm = request.form.get('algorithm')
    params_str = request.form.get('params', '{}')
    if not algorithm or algorithm not in ['kmeans', 'dbscan']:
         return jsonify({"error": "Алгоритм не указан или не поддерживается (ожидается 'kmeans' или 'dbscan')"}), 400
    try:
        params = json.loads(params_str)
        if not isinstance(params, dict): raise ValueError("Params not a dict")
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Invalid params received for user {current_user_id}: {params_str}")
        return jsonify({"error": "Некорректный формат JSON для параметров ('params')"}), 400
    original_filename = file.filename
    if original_filename == '':
        return jsonify({"error": "Имя файла не должно быть пустым"}), 400
    if not allowed_file(original_filename):
        return jsonify({"error": "Недопустимый тип файла. Разрешен только .parquet"}), 400
    file_path = None
    try:
        secure_base_filename = secure_filename(original_filename)
        filename_for_storage = f"{uuid.uuid4()}_{secure_base_filename}"
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_path = os.path.join(upload_folder, filename_for_storage)
        file.save(file_path)
        logger.info(f"User {current_user_id} uploaded file: {original_filename} (saved as {filename_for_storage})")
    except Exception as e:
        logger.error(f"Failed to save uploaded file for user {current_user_id}: {e}", exc_info=True)
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError: pass
        return jsonify({"error": "Не удалось сохранить файл на сервере"}), 500
    try:
        logger.info(f"User {current_user_id} starting SYNC clustering...")
        session_id = run_clustering_pipeline(
            user_id=current_user_id, file_path=file_path, algorithm=algorithm,
            params=params, original_filename=original_filename
        )
        logger.info(f"User {current_user_id} finished SYNC clustering. Session ID: {session_id}")
        return jsonify({"session_id": session_id}), 201
    except (ValueError, SQLAlchemyError) as ve:
        logger.error(f"Validation or DB error during clustering start for user {current_user_id}: {ve}", exc_info=False)
        if file_path and os.path.exists(file_path):
             try: os.remove(file_path)
             except OSError as rem_e: logger.error(f"Error removing file {file_path} after error: {rem_e}")
        return jsonify({"error": f"Ошибка входных данных или БД: {ve}"}), 400
    except Exception as e:
        logger.error(f"SYNC clustering failed unexpectedly for user {current_user_id}: {e}", exc_info=True)
        if file_path and os.path.exists(file_path):
             try: os.remove(file_path)
             except OSError as rem_e: logger.error(f"Error removing file {file_path} after error: {rem_e}")
        return jsonify({"error": "Внутренняя ошибка сервера при кластеризации"}), 500

@clustering_bp.route('/sessions', methods=['GET'])
@jwt_required()
def list_clustering_sessions():
    current_user_id = get_jwt_identity()
    sessions = ClusteringSession.query.filter_by(user_id=current_user_id)\
                                     .order_by(ClusteringSession.created_at.desc()).all()
    output = []
    for session in sessions:
        output.append({
            "session_id": session.id, "created_at": session.created_at.isoformat() + "Z",
            "status": session.status, "algorithm": session.algorithm, "params": session.get_params(),
            "num_clusters": session.num_clusters, "result_message": session.result_message,
            "original_filename": session.original_input_filename if session.original_input_filename else "N/A"
        })
    return jsonify(output), 200

@clustering_bp.route('/results/<session_id>', methods=['GET'])
@jwt_required()
def get_clustering_results(session_id):
    current_user_id = get_jwt_identity()
    session = db.session.get(ClusteringSession, session_id)

    if not session or session.user_id != int(current_user_id):
        return jsonify({"error": "Сессия кластеризации не найдена или не принадлежит вам"}), 404

    base_response = {
        "session_id": session.id, "status": session.status, "algorithm": session.algorithm,
        "params": session.get_params(), "num_clusters": session.num_clusters,
        "processing_time_sec": session.processing_time_sec, "message": session.result_message,
        "original_filename": session.original_input_filename if session.original_input_filename else None,
        "clusters": [],
        "scatter_data": None,
        "scatter_pca_time_sec": None
    }

    is_final_status = session.status in ['SUCCESS', 'RECLUSTERED']
    can_show_partial = session.status == 'PROCESSING'

    cluster_metadatas = session.clusters.filter_by(is_deleted=False).order_by(ClusterMetadata.cluster_label).all()
    num_active_clusters = len(cluster_metadatas)
    base_response["num_clusters"] = num_active_clusters

    clusters_data = []
    for cluster_meta in cluster_metadatas:
        contact_sheet_url = None
        if cluster_meta.contact_sheet_path:
             sheet_filename = os.path.basename(cluster_meta.contact_sheet_path)
             contact_sheet_url = f"/api/clustering/contact_sheet/{session.id}/{sheet_filename}"
        clusters_data.append({
            "id": cluster_meta.cluster_label, "original_id": cluster_meta.original_cluster_id,
            "name": cluster_meta.name, "size": cluster_meta.size,
            "contactSheetUrl": contact_sheet_url, "metrics": cluster_meta.get_metrics(),
            "centroid_2d": cluster_meta.get_centroid_2d()
        })
    base_response["clusters"] = clusters_data

    if session.status == 'PROCESSING' and not base_response.get("message"):
         base_response["message"] = "Идет обработка..."
    elif session.status == 'PROCESSING':
        base_response["message"] = (base_response.get("message", "") + " Идет постобработка...").strip()

    scatter_data_generated = False

    if num_active_clusters == 0 and is_final_status:
        logger.info(f"Нет активных кластеров для сессии {session_id}. Отображение заглушки для PCA.")
        base_response["scatter_data"] = {"message": "Нет активных кластеров для отображения PCA."}
        scatter_data_generated = True

    if not scatter_data_generated and (is_final_status or can_show_partial):
        scatter_cache_file_path = session.scatter_data_file_path
        scatter_data_loaded_from_cache = False

        if scatter_cache_file_path and os.path.exists(scatter_cache_file_path):
            try:
                with open(scatter_cache_file_path, 'r') as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, dict) and "scatter_plot_data" in cached_data and "pca_time" in cached_data:
                     base_response["scatter_data"] = cached_data["scatter_plot_data"]
                     base_response["scatter_pca_time_sec"] = cached_data["pca_time"]
                     scatter_data_loaded_from_cache = True
                     scatter_data_generated = True
                     logger.info(f"Загружены данные Scatter Plot из кэша для сессии {session_id}: {scatter_cache_file_path}")
                else:
                     logger.warning(f"Неверный формат кэша Scatter Plot для сессии {session_id}. Файл будет удален/перезаписан.")
                     session.scatter_data_file_path = None
                     try:
                         os.remove(scatter_cache_file_path)
                     except OSError: pass
                     db.session.commit()
                     scatter_cache_file_path = None

            except (json.JSONDecodeError, OSError, IOError) as e:
                logger.error(f"Ошибка загрузки кэша Scatter Plot ({scatter_cache_file_path}) для сессии {session_id}: {e}. Будет перегенерация.", exc_info=True)
                session.scatter_data_file_path = None
                try:
                    if os.path.exists(scatter_cache_file_path): os.remove(scatter_cache_file_path)
                except OSError: pass
                db.session.commit()
                scatter_cache_file_path = None

        if not scatter_data_loaded_from_cache and is_final_status:
            if session.input_file_path and os.path.exists(session.input_file_path):
                logger.info(f"Генерация данных Scatter Plot для сессии {session_id} (кэш не найден/невалиден/пропущен)")
                try:
                    embeddings, _, _ = load_embeddings(session.input_file_path)
                    labels = get_cluster_labels_for_session(session, embeddings)

                    if embeddings is not None and labels is not None:
                        new_cache_path, pca_time = generate_and_save_scatter_data(session.id, embeddings, labels)

                        if new_cache_path:
                            try:
                                with open(new_cache_path, 'r') as f:
                                    new_cache_content = json.load(f)
                                base_response["scatter_data"] = new_cache_content.get("scatter_plot_data")
                                base_response["scatter_pca_time_sec"] = new_cache_content.get("pca_time")
                                scatter_data_generated = True
                                session.scatter_data_file_path = new_cache_path
                                flag_modified(session, "scatter_data_file_path")
                                db.session.commit()
                                logger.info(f"Scatter plot сгенерирован, сохранен и загружен для ответа сессии {session_id}")
                            except (OSError, IOError, json.JSONDecodeError) as read_err:
                                logger.error(f"Не удалось прочитать свежесозданный кэш scatter plot {new_cache_path}: {read_err}")
                                base_response["scatter_data"] = {"error": "Ошибка чтения сгенерированных данных PCA."}
                                scatter_data_generated = True
                        else:
                             logger.error(f"Функция generate_and_save_scatter_data не смогла создать кэш для сессии {session_id}")
                             base_response["scatter_data"] = {"error": "Ошибка генерации или сохранения данных PCA."}
                             scatter_data_generated = True
                    else:
                        logger.warning(f"Не удалось получить эмбеддинги или метки для генерации scatter plot, сессия {session_id}")
                        base_response["scatter_data"] = {"error": "Не удалось загрузить данные для scatter plot."}
                        scatter_data_generated = True

                except Exception as e:
                    logger.error(f"Ошибка генерации данных scatter plot для сессии {session_id}: {e}", exc_info=True)
                    base_response["scatter_data"] = {"error": "Внутренняя ошибка при генерации данных scatter plot."}
                    scatter_data_generated = True
            else:
                 logger.warning(f"Путь к входному файлу не найден для сессии {session_id}, невозможно сгенерировать scatter plot.")
                 base_response["scatter_data"] = {"error": "Файл с входными данными не найден."}
                 scatter_data_generated = True

    if base_response["scatter_data"] is None and not scatter_data_generated:
         if not is_final_status and not can_show_partial:
            base_response["scatter_data"] = {"message": f"Визуализация PCA недоступна для статуса '{session.status}'."}
         elif can_show_partial:
             base_response["scatter_data"] = {"message": "Генерация данных PCA начнется после завершения кластеризации."}

    if not is_final_status and not can_show_partial and not base_response.get("error"):
        base_response["error"] = f"Статус сессии: {session.status}. {session.result_message or ''}"

    return jsonify(base_response), 200

@clustering_bp.route('/contact_sheet/<session_id>/<filename>', methods=['GET'])
@jwt_required()
def get_contact_sheet_image(session_id, filename):
    current_user_id = get_jwt_identity()
    session = db.session.get(ClusteringSession, session_id)
    if not session or session.user_id != int(current_user_id):
        return jsonify({"error": "Сессия не найдена или доступ запрещен"}), 404
    if not filename.startswith(f"cs_{session_id}_") or not filename.lower().endswith(f".{current_app.config.get('CONTACT_SHEET_OUTPUT_FORMAT', 'JPEG').lower()}"):
        logger.warning(f"Attempt to access invalid contact sheet filename: {filename} for session {session_id}")
        return jsonify({"error": "Некорректное имя файла отпечатка"}), 400
    contact_sheet_session_dir = os.path.join(current_app.config['CONTACT_SHEET_FOLDER'], session_id)
    safe_filename = secure_filename(filename)
    try:
        mimetype = f"image/{current_app.config.get('CONTACT_SHEET_OUTPUT_FORMAT', 'JPEG').lower()}"
        return send_from_directory(contact_sheet_session_dir, safe_filename, mimetype=mimetype)
    except FileNotFoundError:
         logger.warning(f"Contact sheet file not found: session={session_id}, filename={safe_filename}")
         return jsonify({"error": "Файл контактного отпечатка не найден"}), 404

@clustering_bp.route('/results/<session_id>/cluster/<cluster_label>', methods=['DELETE'])
@jwt_required()
def delete_and_redistribute_cluster(session_id, cluster_label):
    current_user_id = get_jwt_identity()
    logger.info(f"User {current_user_id} requested DELETE/REDISTRIBUTE for cluster {cluster_label} in session {session_id}")

    try:
        result = redistribute_cluster_data(
            session_id=session_id,
            cluster_label_to_remove_str=cluster_label,
            user_id=int(current_user_id)
        )
        logger.info(f"User {current_user_id} finished DELETE/REDISTRIBUTE for {session_id}/{cluster_label}. Result: {result}")
        return jsonify(result), 200

    except ValueError as ve:
         logger.warning(f"Validation error during redistribute for {session_id}/{cluster_label} by user {current_user_id}: {ve}", exc_info=False)
         return jsonify({"error": f"Ошибка операции: {ve}"}), 400
    except RuntimeError as re:
         logger.error(f"Runtime error during redistribute for {session_id}/{cluster_label} by user {current_user_id}: {re}", exc_info=True)
         return jsonify({"error": f"Внутренняя ошибка сервера при перераспределении: {re}"}), 500
    except SQLAlchemyError as e:
         logger.error(f"DB error during redistribute {session_id}/{cluster_label} by user {current_user_id}: {e}", exc_info=True)
         return jsonify({"error": "Ошибка базы данных при выполнении операции"}), 500
    except Exception as e:
        logger.error(f"Unexpected error during redistribute for {session_id}/{cluster_label} by user {current_user_id}: {e}", exc_info=True)
        return jsonify({"error": "Неизвестная внутренняя ошибка сервера"}), 500


@clustering_bp.route('/results/<session_id>/adjust', methods=['POST'])
@jwt_required()
def adjust_clusters(session_id):
    current_user_id = get_jwt_identity()
    session = db.session.get(ClusteringSession, session_id)
    if not session or session.user_id != int(current_user_id):
        return jsonify({"error": "Сессия не найдена или доступ запрещен"}), 404
    if session.status not in ['SUCCESS', 'RECLUSTERED']:
        return jsonify({"error": f"Невозможно редактировать сессию со статусом {session.status}"}), 400
    data = request.get_json()
    action = data.get('action')
    cluster_id = data.get('cluster_id')
    new_name = data.get('new_name')
    if not action or action != 'RENAME' or not cluster_id or new_name is None:
        return jsonify({"error": "Некорректные параметры для 'RENAME' (нужны action='RENAME', cluster_id, new_name)"}), 400
    cluster_to_rename = session.clusters.filter_by(cluster_label=str(cluster_id), is_deleted=False).first()
    if not cluster_to_rename:
         return jsonify({"error": f"Кластер с ID {cluster_id} не найден или удален"}), 404
    old_name = cluster_to_rename.name
    cluster_to_rename.name = new_name if new_name else None
    log_manual_adjustment(session.id, current_user_id, "RENAME", {
        "cluster_label": cluster_id, "old_name": old_name, "new_name": cluster_to_rename.name
    })
    try:
        db.session.commit()
        logger.info(f"User {current_user_id} renamed cluster {cluster_id} in session {session_id} to '{cluster_to_rename.name}'")
        return jsonify({"message": "Кластер переименован", "cluster": {"id": cluster_to_rename.cluster_label, "name": cluster_to_rename.name, }}), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"DB error renaming cluster {session_id}/{cluster_id}: {e}", exc_info=True)
        return jsonify({"error": "Ошибка БД при переименовании кластера"}), 500

def register_clustering_routes(app):
    app.register_blueprint(clustering_bp)