import os
import uuid
import json
import logging
import time
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np
from sklearn.decomposition import PCA
from werkzeug.utils import secure_filename
from models import db, ClusteringSession, ClusterMetadata
from sqlalchemy.exc import SQLAlchemyError
from clustering_logic import (
    run_clustering_pipeline, run_reclustering_pipeline, log_manual_adjustment,
    load_embeddings, get_cluster_labels_for_session
)
import config

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

    clusters_data = []
    cluster_metadatas = session.clusters.filter_by(is_deleted=False).order_by(ClusterMetadata.cluster_label).all()
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
    base_response["num_clusters"] = len(clusters_data)

    if session.status == 'PROCESSING' and not base_response.get("message"):
         base_response["message"] = "Идет обработка..."
    elif session.status == 'PROCESSING':
        base_response["message"] = (base_response.get("message", "") + " Идет постобработка...").strip()

    if is_final_status:
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
                     logger.info(f"Загружены данные Scatter Plot из кэша для сессии {session_id}: {scatter_cache_file_path}")
                else:
                     logger.warning(f"Неверный формат кэша Scatter Plot для сессии {session_id}. Файл будет перезаписан.")
                     session.scatter_data_file_path = None
                     db.session.commit()
                     scatter_cache_file_path = None

            except (json.JSONDecodeError, OSError, IOError) as e:
                logger.error(f"Ошибка загрузки кэша Scatter Plot ({scatter_cache_file_path}) для сессии {session_id}: {e}. Будет перегенерация.", exc_info=True)
                session.scatter_data_file_path = None
                db.session.commit()
                scatter_cache_file_path = None

        if not scatter_data_loaded_from_cache:
            if session.input_file_path and os.path.exists(session.input_file_path):
                logger.info(f"Генерация данных Scatter Plot для сессии {session_id} (кэш не найден или невалиден)")
                pca_start_time = time.time()
                try:
                    embeddings, _, _ = load_embeddings(session.input_file_path)
                    labels = get_cluster_labels_for_session(session, embeddings)

                    if embeddings is not None and labels is not None and embeddings.shape[0] == labels.shape[0]:
                        num_points = embeddings.shape[0]
                        max_points = config.Config.MAX_SCATTER_PLOT_POINTS
                        indices = np.arange(num_points)

                        if num_points > max_points:
                            logger.warning(f"Session {session_id}: Слишком много точек ({num_points}) для scatter plot, сэмплирование до {max_points}")
                            indices = np.random.choice(indices, max_points, replace=False)
                            embeddings_sampled = embeddings[indices]
                            labels_sampled = labels[indices]
                        else:
                            embeddings_sampled = embeddings
                            labels_sampled = labels

                        if embeddings_sampled.shape[0] >= 2:
                            pca = PCA(n_components=2, svd_solver='full', random_state=42)
                            try:
                                embeddings_2d = pca.fit_transform(embeddings_sampled)
                                pca_end_time = time.time()
                                pca_time_sec = pca_end_time - pca_start_time
                                base_response["scatter_pca_time_sec"] = pca_time_sec

                                scatter_plot_data = []
                                for i in range(embeddings_2d.shape[0]):
                                    scatter_plot_data.append({
                                        'x': float(embeddings_2d[i, 0]),
                                        'y': float(embeddings_2d[i, 1]),
                                        'cluster': str(labels_sampled[i])
                                    })
                                base_response["scatter_data"] = scatter_plot_data
                                logger.info(f"Успешно сгенерированы данные Scatter Plot для сессии {session_id} ({len(scatter_plot_data)} точек) за {pca_time_sec:.2f} сек.")

                                try:
                                    scatter_filename = f"scatter_{session_id}.json"
                                    scatter_cache_file_path = os.path.join(current_app.config['SCATTER_DATA_FOLDER'], scatter_filename)
                                    cache_content = {
                                        "scatter_plot_data": scatter_plot_data,
                                        "pca_time": pca_time_sec
                                    }
                                    with open(scatter_cache_file_path, 'w') as f:
                                        json.dump(cache_content, f)

                                    session.scatter_data_file_path = scatter_cache_file_path
                                    db.session.commit()
                                    logger.info(f"Данные Scatter Plot сохранены в кэш для сессии {session_id}: {scatter_cache_file_path}")

                                except (OSError, IOError) as e:
                                    db.session.rollback()
                                    logger.error(f"Не удалось сохранить кэш Scatter Plot для сессии {session_id}: {e}", exc_info=True)
                                except SQLAlchemyError as e_db:
                                    db.session.rollback()
                                    logger.error(f"Не удалось обновить путь кэша в БД для сессии {session_id}: {e_db}", exc_info=True)

                            except ValueError as pca_err:
                                logger.error(f"PCA не удался для scatter plot в сессии {session_id}: {pca_err}", exc_info=True)
                                base_response["scatter_data"] = {"error": f"Ошибка расчета PCA: {pca_err}"}
                        else:
                             logger.warning(f"Недостаточно точек ({embeddings_sampled.shape[0]}) после сэмплирования для PCA в сессии {session_id}")
                             base_response["scatter_data"] = {"error": "Недостаточно данных для визуализации после сэмплирования."}
                    else:
                        logger.warning(f"Не удалось получить эмбеддинги или метки для scatter plot, сессия {session_id}")
                        base_response["scatter_data"] = {"error": "Не удалось загрузить эмбеддинги или метки для scatter plot."}

                except Exception as e:
                    logger.error(f"Ошибка генерации данных scatter plot для сессии {session_id}: {e}", exc_info=True)
                    base_response["scatter_data"] = {"error": "Внутренняя ошибка при генерации данных scatter plot."}
            else:
                 logger.warning(f"Путь к входному файлу не найден или некорректен для сессии {session_id}, невозможно сгенерировать scatter plot.")
                 base_response["scatter_data"] = {"error": "Файл с входными данными не найден."}

    elif not is_final_status and not can_show_partial:
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
def delete_cluster_and_recluster(session_id, cluster_label):
    current_user_id = get_jwt_identity()
    session = db.session.get(ClusteringSession, session_id)
    if not session or session.user_id != int(current_user_id):
        return jsonify({"error": "Сессия не найдена или доступ запрещен"}), 404
    if session.status not in ['SUCCESS', 'RECLUSTERED']:
        return jsonify({"error": f"Невозможно удалить кластер из сессии со статусом {session.status}"}), 400
    cluster_to_delete = session.clusters.filter_by(cluster_label=cluster_label, is_deleted=False).first()
    if not cluster_to_delete:
         return jsonify({"error": "Кластер для удаления не найден или уже удален"}), 404
    cluster_to_delete.is_deleted = True
    sheet_path = cluster_to_delete.contact_sheet_path
    cluster_to_delete.contact_sheet_path = None
    log_manual_adjustment(session.id, current_user_id, "DELETE_CLUSTER", {"cluster_label": cluster_label})
    try:
        db.session.commit()
        logger.info(f"User {current_user_id} marked cluster {cluster_label} in session {session_id} for deletion.")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"DB error marking cluster for deletion {session_id}/{cluster_label}: {e}", exc_info=True)
        return jsonify({"error": "Ошибка БД при удалении кластера"}), 500
    if sheet_path and os.path.exists(sheet_path):
        try:
            os.remove(sheet_path)
            logger.info(f"Deleted contact sheet file: {sheet_path}")
        except OSError as e:
             logger.error(f"Error deleting contact sheet file {sheet_path}: {e}")
    try:
        logger.info(f"User {current_user_id} starting SYNC re-clustering for session {session_id}...")
        new_session_id = run_reclustering_pipeline(
            original_session_id=session_id, user_id=current_user_id
        )
        logger.info(f"User {current_user_id} finished SYNC re-clustering. New session ID: {new_session_id}")
        return jsonify({"message": "Кластер удален, рекластеризация завершена.", "new_session_id": new_session_id}), 200
    except (ValueError, SQLAlchemyError) as ve:
         logger.error(f"Validation or DB error during re-clustering start for session {session_id}: {ve}", exc_info=False)
         return jsonify({"error": f"Ошибка входных данных или БД при рекластеризации: {ve}"}), 400
    except Exception as e:
        logger.error(f"SYNC re-clustering failed unexpectedly for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": "Внутренняя ошибка сервера при рекластеризации"}), 500

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