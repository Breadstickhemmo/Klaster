import os
import time
import logging
import numpy as np
import pandas as pd
import json
import uuid
from flask import current_app
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, euclidean_distances
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("Faiss не найден. Поиск ближайших соседей будет медленным.")

from PIL import Image, ImageDraw, ImageFont
from models import db, ClusteringSession, ClusterMetadata, ManualAdjustmentLog
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.exc import SQLAlchemyError
import config

logger = logging.getLogger(__name__)

def load_embeddings(file_path):
    try:
        df = pd.read_parquet(file_path)
        if 'embedding' not in df.columns:
            embedding_cols = df.select_dtypes(include=np.number).columns
            if len(embedding_cols) > 1:
                 embeddings = df[embedding_cols].values.astype(np.float32)
            else:
                 raise ValueError("Столбец 'embedding' или числовые столбцы не найдены в Parquet файле")
        else:
             embeddings = np.array(df['embedding'].tolist(), dtype=np.float32)

        image_ids = df['id'].astype(str).tolist() if 'id' in df.columns else [str(i) for i in df.index.tolist()]
        image_paths = df['image_path'].tolist() if 'image_path' in df.columns else None

        logger.info(f"Загружено {embeddings.shape[0]} эмбеддингов из {file_path}")
        return embeddings, image_ids, image_paths

    except Exception as e:
        logger.error(f"Ошибка загрузки Parquet файла {file_path}: {e}", exc_info=True)
        raise

def perform_clustering(embeddings, algorithm, params):
    n_samples = embeddings.shape[0]
    start_time = time.time()
    logger.info(f"Запуск кластеризации: алгоритм={algorithm}, параметры={params}, размер данных={n_samples}")

    if algorithm == 'kmeans':
        n_clusters = int(params.get('n_clusters', 5))
        if n_clusters <= 0: raise ValueError("n_clusters должен быть > 0 для K-means")
        logger.info(f"Используем K-means с n_clusters={n_clusters}")
        if FAISS_AVAILABLE and n_samples > 10000:
            logger.info("Используем Faiss K-means")
            d = embeddings.shape[1]
            kmeans = faiss.Kmeans(d=d, k=n_clusters, niter=20, verbose=False, gpu=False)
            kmeans.train(embeddings)
            _, labels = kmeans.index.search(embeddings, 1)
            labels = labels.flatten()
            centroids = kmeans.centroids
        else:
             logger.info("Используем Scikit-learn K-means")
             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
             labels = kmeans.fit_predict(embeddings)
             centroids = kmeans.cluster_centers_

    elif algorithm == 'dbscan':
        eps = float(params.get('eps', 0.5))
        min_samples = int(params.get('min_samples', 5))
        if eps <= 0 or min_samples <=0 : raise ValueError("eps и min_samples должны быть > 0 для DBSCAN")
        logger.info(f"Используем DBSCAN с eps={eps}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(embeddings)
        unique_labels = np.unique(labels)
        centroids = []
        actual_labels = []
        for label in unique_labels:
            if label == -1: continue
            cluster_points = embeddings[labels == label]
            if cluster_points.shape[0] > 0:
                 centroid = np.mean(cluster_points, axis=0)
                 centroids.append(centroid)
                 actual_labels.append(label)
            else:
                 logger.warning(f"DBSCAN: Кластер {label} не содержит точек, пропуск.")
        centroids = np.array(centroids) if centroids else np.empty((0, embeddings.shape[1]))
        new_labels = np.full_like(labels, -1)
        label_map = {old_label: i for i, old_label in enumerate(actual_labels)}
        for old_label, new_label_index in label_map.items():
             new_labels[labels == old_label] = new_label_index
        labels = new_labels

    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

    processing_time = time.time() - start_time
    logger.info(f"Кластеризация завершена за {processing_time:.2f} сек.")
    metrics = {}

    return labels, centroids, metrics, processing_time

def find_nearest_images_to_centroids(embeddings, labels, centroids, image_ids, image_paths, n_images):
    nearest_neighbors = {}
    if not image_paths or centroids.shape[0] == 0: return nearest_neighbors

    unique_labels = np.unique(labels)
    num_clusters = centroids.shape[0]

    faiss_index = None
    if FAISS_AVAILABLE:
        try:
            d = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(d)
            faiss_index.add(embeddings)
        except Exception as faiss_e:
            logger.error(f"Ошибка создания Faiss индекса: {faiss_e}")
            faiss_index = None

    for i in range(num_clusters):
        current_label = i
        cluster_mask = (labels == current_label)
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0: continue
        cluster_embeddings = embeddings[cluster_indices]
        centroid = centroids[i]
        k_search = min(n_images, len(cluster_indices))
        if k_search == 0: continue
        distances = []
        indices_in_cluster = []

        if faiss_index:
            try:
                 D, I = faiss_index.search(np.array([centroid]), k=min(k_search * 5, embeddings.shape[0]))
                 indices_global = I[0]
                 filtered_indices = [idx for idx in indices_global if labels[idx] == current_label]
                 top_k_indices_global = filtered_indices[:k_search]
                 final_distances = []
                 for global_idx in top_k_indices_global:
                      dist = np.linalg.norm(embeddings[global_idx] - centroid)
                      final_distances.append(dist)

                 indices_local = []
                 cluster_indices_list = list(cluster_indices)
                 map_global_to_local = {glob_idx: loc_idx for loc_idx, glob_idx in enumerate(cluster_indices_list)}

                 for glob_idx in top_k_indices_global:
                     if glob_idx in map_global_to_local:
                         indices_local.append(map_global_to_local[glob_idx])
                     else:
                         logger.warning(f"Global index {glob_idx} not found in cluster_indices for cluster {current_label} during Faiss post-processing.")


                 distances = np.array(final_distances)
                 indices_in_cluster = np.array(indices_local)
            except Exception as faiss_search_e:
                logger.warning(f"Ошибка поиска Faiss для кластера {current_label}: {faiss_search_e}. Переключение на sklearn.")
                faiss_index = None

        if not faiss_index:
             dist_to_centroid = euclidean_distances(cluster_embeddings, np.array([centroid])).flatten()
             sorted_indices_local = np.argsort(dist_to_centroid)
             indices_in_cluster = sorted_indices_local[:k_search]
             distances = dist_to_centroid[indices_in_cluster]

        neighbors_for_cluster = []
        for j, local_idx in enumerate(indices_in_cluster):
            global_idx = cluster_indices[local_idx]
            img_id = image_ids[global_idx]
            img_path = image_paths[global_idx]
            dist = distances[j]
            neighbors_for_cluster.append((img_id, img_path, float(dist)))
        nearest_neighbors[current_label] = neighbors_for_cluster
    return nearest_neighbors

def create_contact_sheet(image_paths, output_path, grid_size, thumb_size, format='JPEG'):
    if not image_paths: return False
    cols, rows = grid_size
    thumb_w, thumb_h = thumb_size
    gap = 5
    total_width = cols * thumb_w + (cols + 1) * gap
    total_height = rows * thumb_h + (rows + 1) * gap
    contact_sheet = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(contact_sheet)
    try: font = ImageFont.truetype("arial.ttf", 10)
    except IOError: font = ImageFont.load_default()
    current_col, current_row = 0, 0
    for img_path in image_paths[:cols*rows]:
        try:
            img = Image.open(img_path)
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            x_pos = gap + current_col * (thumb_w + gap)
            y_pos = gap + current_row * (thumb_h + gap)
            contact_sheet.paste(img, (x_pos, y_pos))
        except FileNotFoundError:
            logger.warning(f"Файл не найден для отпечатка: {img_path}")
            x_pos = gap + current_col * (thumb_w + gap); y_pos = gap + current_row * (thumb_h + gap)
            draw.rectangle([x_pos, y_pos, x_pos + thumb_w, y_pos + thumb_h], fill="lightgray", outline="red")
            draw.text((x_pos + 5, y_pos + 5), "Not Found", fill="red", font=font)
        except Exception as e:
            logger.error(f"Ошибка обработки {img_path} для отпечатка: {e}")
            x_pos = gap + current_col * (thumb_w + gap); y_pos = gap + current_row * (thumb_h + gap)
            draw.rectangle([x_pos, y_pos, x_pos + thumb_w, y_pos + thumb_h], fill="lightgray", outline="orange")
            draw.text((x_pos + 5, y_pos + 5), "Error", fill="orange", font=font)
        current_col += 1
        if current_col >= cols:
            current_col = 0; current_row += 1
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        contact_sheet.save(output_path, format=format, quality=85)
        logger.info(f"Контактный отпечаток сохранен: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения контактного отпечатка {output_path}: {e}", exc_info=True)
        return False

def calculate_and_save_centroids_2d(session_id):
    logger.info(f"Calculating 2D centroids for session {session_id}")
    try:
        clusters = ClusterMetadata.query.filter_by(session_id=session_id, is_deleted=False).all()
        if not clusters:
            logger.info(f"No active clusters found for session {session_id} to calculate 2D centroids.")
            all_clusters_in_session = ClusterMetadata.query.filter_by(session_id=session_id).all()
            for c_meta in all_clusters_in_session:
                 if c_meta.centroid_2d_json is not None:
                    c_meta.set_centroid_2d(None)
                    flag_modified(c_meta, "centroid_2d_json")
            db.session.commit()
            return

        original_centroids = []
        cluster_map = {}
        valid_indices = []
        for i, cluster_meta in enumerate(clusters):
            centroid_vec = cluster_meta.get_centroid()
            if centroid_vec is not None:
                original_centroids.append(centroid_vec)
                cluster_map[len(original_centroids) - 1] = cluster_meta.id
                valid_indices.append(i)
            else:
                 logger.warning(f"Centroid vector is None for cluster {cluster_meta.id} in session {session_id}")

        if len(original_centroids) < 2:
            logger.warning(f"Need at least 2 valid centroids for PCA, found {len(original_centroids)} for session {session_id}. Skipping 2D calculation.")
            for cluster_meta in clusters:
                cluster_meta.set_centroid_2d(None)
                flag_modified(cluster_meta, "centroid_2d_json")
            db.session.commit()
            return

        original_centroids_np = np.array(original_centroids)
        pca = PCA(n_components=2, svd_solver='full', random_state=42)
        try:
            centroids_2d = pca.fit_transform(original_centroids_np)
        except ValueError as pca_err:
             logger.error(f"PCA failed for session {session_id}: {pca_err}", exc_info=True)
             for cluster_meta in clusters:
                 cluster_meta.set_centroid_2d(None)
                 flag_modified(cluster_meta, "centroid_2d_json")
             db.session.commit()
             return

        for idx_in_pca, cluster_db_id in cluster_map.items():
            cluster_to_update = next((c for c in clusters if c.id == cluster_db_id), None)
            if cluster_to_update:
                coords_2d = centroids_2d[idx_in_pca]
                cluster_to_update.set_centroid_2d(coords_2d)
                logger.debug(f"Saved 2D centroid for cluster {cluster_to_update.id}: {coords_2d}")
                flag_modified(cluster_to_update, "centroid_2d_json")

        all_clusters_in_session = ClusterMetadata.query.filter_by(session_id=session_id).all()
        updated_cluster_ids = set(cluster_map.values())
        for c_meta in all_clusters_in_session:
            if c_meta.id not in updated_cluster_ids and c_meta.centroid_2d_json is not None:
                 c_meta.set_centroid_2d(None)
                 flag_modified(c_meta, "centroid_2d_json")

        db.session.commit()
        logger.info(f"Successfully calculated and saved 2D centroids for session {session_id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error calculating/saving 2D centroids for session {session_id}: {e}", exc_info=True)

def save_clustering_results(session, labels, centroids, nearest_neighbors_map, image_paths_available):
    app_config = current_app.config
    contact_sheet_dir_base = app_config['CONTACT_SHEET_FOLDER']
    grid_size = app_config.get('CONTACT_SHEET_GRID_SIZE', (3, 3))
    thumb_size = app_config.get('CONTACT_SHEET_THUMBNAIL_SIZE', (100, 100))
    output_format = app_config.get('CONTACT_SHEET_OUTPUT_FORMAT', 'JPEG')
    unique_labels = np.unique(labels)
    created_cluster_metadata = []

    for i, label in enumerate(unique_labels):
        if label == -1: continue
        cluster_mask = (labels == label)
        cluster_size = int(np.sum(cluster_mask))
        centroid_index = int(label)
        if centroid_index < 0 or centroid_index >= len(centroids):
             logger.error(f"Некорректный индекс центроида {centroid_index} для метки {label} в сессии {session.id}")
             continue
        centroid_vector = centroids[centroid_index]
        cluster_meta = ClusterMetadata(
            session_id=session.id, cluster_label=str(label), original_cluster_id=str(label),
            size=cluster_size, is_deleted=False
        )
        cluster_meta.set_centroid(centroid_vector)
        cluster_meta.set_metrics({})
        cluster_meta.set_centroid_2d(None)
        contact_sheet_path = None
        if image_paths_available and label in nearest_neighbors_map and nearest_neighbors_map[label]:
            sheet_filename = f"cs_{session.id}_{label}.{output_format.lower()}"
            session_sheet_dir = os.path.join(contact_sheet_dir_base, session.id)
            sheet_full_path = os.path.join(session_sheet_dir, sheet_filename)
            neighbor_paths = [p for _, p, _ in nearest_neighbors_map[label]]
            if create_contact_sheet(neighbor_paths, sheet_full_path, grid_size, thumb_size, output_format):
                 contact_sheet_path = sheet_full_path
        cluster_meta.contact_sheet_path = contact_sheet_path
        db.session.add(cluster_meta)
        created_cluster_metadata.append(cluster_meta)
    return created_cluster_metadata

def get_cluster_labels_for_session(session: ClusteringSession, embeddings: np.ndarray) -> np.ndarray | None:
    if not session or embeddings is None:
        return None

    algorithm = session.algorithm
    params = session.get_params()
    logger.info(f"Re-calculating labels for session {session.id} using {algorithm} with params {params}")

    try:
        if algorithm == 'kmeans':
            n_clusters_param = int(params.get('n_clusters', 5))
            if n_clusters_param <= 0: raise ValueError("n_clusters > 0 required")

            active_clusters = session.clusters.filter_by(is_deleted=False).all()
            stored_centroids = []
            label_map_from_db = {}
            for cluster_meta in active_clusters:
                 centroid_vec = cluster_meta.get_centroid()
                 if centroid_vec is not None and centroid_vec.shape[0] == embeddings.shape[1]:
                     stored_centroids.append(centroid_vec)
                     try:
                         label_map_from_db[int(cluster_meta.cluster_label)] = len(stored_centroids) - 1
                     except ValueError:
                          logger.warning(f"Could not parse cluster label {cluster_meta.cluster_label} to int for K-means assignment.")
                          continue

            if len(stored_centroids) == n_clusters_param:
                logger.info(f"Using {len(stored_centroids)} stored centroids for label assignment in session {session.id}")
                stored_centroids_np = np.array(stored_centroids)
                distances = euclidean_distances(embeddings, stored_centroids_np)
                assigned_centroid_indices = np.argmin(distances, axis=1)

                reverse_label_map = {v: k for k, v in label_map_from_db.items()}
                labels = np.array([reverse_label_map.get(idx, -1) for idx in assigned_centroid_indices])
                if -1 in labels:
                     logger.warning(f"Some points could not be assigned to stored K-means centroids in session {session.id}")

            elif len(stored_centroids) > 0 and len(stored_centroids) != n_clusters_param:
                 logger.warning(f"Stored active centroid count ({len(stored_centroids)}) differs from n_clusters param ({n_clusters_param}) for session {session.id}. Using stored centroids for assignment.")
                 stored_centroids_np = np.array(stored_centroids)
                 distances = euclidean_distances(embeddings, stored_centroids_np)
                 assigned_centroid_indices = np.argmin(distances, axis=1)
                 reverse_label_map = {v: k for k, v in label_map_from_db.items()}
                 labels = np.array([reverse_label_map.get(idx, -1) for idx in assigned_centroid_indices])
                 if -1 in labels:
                     logger.warning(f"Some points could not be assigned to stored K-means centroids in session {session.id}")

            else:
                logger.warning(f"No valid stored centroids found or count mismatch for session {session.id}. Re-running KMeans fit_predict with n_clusters={n_clusters_param}.")
                kmeans = KMeans(n_clusters=n_clusters_param, random_state=42, n_init=1)
                labels = kmeans.fit_predict(embeddings)


        elif algorithm == 'dbscan':
            eps = float(params.get('eps', 0.5))
            min_samples = int(params.get('min_samples', 5))
            if eps <= 0 or min_samples <=0 : raise ValueError("eps and min_samples > 0 required")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            raw_labels = dbscan.fit_predict(embeddings)

            unique_raw_labels = np.unique(raw_labels)
            actual_labels = [lbl for lbl in unique_raw_labels if lbl != -1]
            new_labels = np.full_like(raw_labels, -1)
            label_map = {old_label: i for i, old_label in enumerate(actual_labels)}
            for old_label, new_label_index in label_map.items():
                 new_labels[raw_labels == old_label] = new_label_index
            labels = new_labels

        else:
            logger.error(f"Unknown algorithm '{algorithm}' found for session {session.id}")
            return None

        logger.info(f"Successfully re-calculated labels for session {session.id}")
        return labels

    except Exception as e:
        logger.error(f"Error re-calculating labels for session {session.id}: {e}", exc_info=True)
        return None

def run_clustering_pipeline(user_id, file_path, algorithm, params, original_filename):
    session_id = str(uuid.uuid4())
    logger.info(f"Запуск синхронной кластеризации {session_id} для пользователя {user_id}")
    session = ClusteringSession(
        id=session_id, user_id=user_id, status='STARTED', algorithm=algorithm,
        input_file_path=file_path, original_input_filename=original_filename
    )
    session.set_params(params)
    db.session.add(session)
    try:
        db.session.commit()
    except SQLAlchemyError as commit_err:
        db.session.rollback()
        logger.error(f"Ошибка создания сессии {session_id}: {commit_err}", exc_info=True)
        raise ValueError(f"Не удалось создать сессию в БД: {commit_err}") from commit_err

    try:
        embeddings, image_ids, image_paths = load_embeddings(file_path)
        image_paths_available = image_paths is not None
        labels, centroids, metrics, processing_time = perform_clustering(embeddings, algorithm, params)
        session.processing_time_sec = processing_time
        unique_labels = np.unique(labels)
        n_clusters_found = len(centroids)
        session.num_clusters = n_clusters_found
        logger.info(f"Session {session_id}: Найдено {n_clusters_found} кластеров.")
        nearest_neighbors_map = {}
        if image_paths_available:
            logger.info(f"Session {session_id}: Поиск изображений для отпечатков...")
            images_per_cluster = current_app.config.get('CONTACT_SHEET_IMAGES_PER_CLUSTER', 9)
            nearest_neighbors_map = find_nearest_images_to_centroids(
                embeddings, labels, centroids, image_ids, image_paths, images_per_cluster
            )
        logger.info(f"Session {session_id}: Сохранение результатов кластеризации...")
        save_clustering_results(session, labels, centroids, nearest_neighbors_map, image_paths_available)
        session.status = 'PROCESSING'
        db.session.commit()
        calculate_and_save_centroids_2d(session.id)
        session.status = 'SUCCESS'
        session.result_message = f'Кластеризация завершена. Найдено {n_clusters_found} кластеров.'
        db.session.commit()
        logger.info(f"Синхронная кластеризация {session_id} успешно завершена (включая 2D центроиды).")
        return session.id
    except Exception as e:
        db.session.rollback()
        logger.error(f"Ошибка в синхронной кластеризации {session_id}: {e}", exc_info=True)
        try:
             fail_session = db.session.get(ClusteringSession, session_id)
             if fail_session:
                fail_session.status = 'FAILURE'
                fail_session.result_message = f"Ошибка: {str(e)[:500]}"
                db.session.commit()
             else:
                logger.error(f"Не удалось найти сессию {session_id} для обновления статуса на FAILURE")
        except Exception as db_err:
             db.session.rollback()
             logger.error(f"Не удалось обновить статус сессии {session_id} на FAILURE: {db_err}", exc_info=True)
        raise

def run_reclustering_pipeline(original_session_id, user_id):
    new_session_id = str(uuid.uuid4())
    logger.info(f"Запуск синхронной рекластеризации {new_session_id} для сессии {original_session_id}")
    original_session = db.session.get(ClusteringSession, original_session_id)
    if not original_session or original_session.user_id != user_id:
        raise ValueError("Исходная сессия не найдена или доступ запрещен")
    new_session = ClusteringSession(
        id=new_session_id, user_id=user_id, status='STARTED', algorithm=original_session.algorithm,
        input_file_path=original_session.input_file_path, original_input_filename=original_session.original_input_filename
    )
    new_session.set_params(original_session.get_params())
    db.session.add(new_session)
    original_session.status = 'RECLUSTERING_STARTED'
    try:
        db.session.commit()
    except SQLAlchemyError as commit_err:
        db.session.rollback()
        logger.error(f"Ошибка создания сессии рекластеризации {new_session_id}: {commit_err}", exc_info=True)
        raise ValueError(f"Не удалось создать сессию рекластеризации: {commit_err}") from commit_err

    try:
        embeddings, image_ids, image_paths = load_embeddings(original_session.input_file_path)
        image_paths_available = image_paths is not None
        labels, centroids, metrics, processing_time = perform_clustering(
            embeddings, new_session.algorithm, new_session.get_params()
        )
        new_session.processing_time_sec = processing_time
        unique_labels = np.unique(labels)
        n_clusters_found = len(centroids)
        new_session.num_clusters = n_clusters_found
        logger.info(f"Recluster {new_session_id}: Найдено {n_clusters_found} кластеров.")
        nearest_neighbors_map = {}
        if image_paths_available:
            logger.info(f"Recluster {new_session_id}: Поиск изображений...")
            images_per_cluster = current_app.config.get('CONTACT_SHEET_IMAGES_PER_CLUSTER', 9)
            nearest_neighbors_map = find_nearest_images_to_centroids(
                embeddings, labels, centroids, image_ids, image_paths, images_per_cluster
            )
        logger.info(f"Recluster {new_session_id}: Сохранение результатов...")
        save_clustering_results(new_session, labels, centroids, nearest_neighbors_map, image_paths_available)
        new_session.status = 'PROCESSING'
        db.session.commit()
        calculate_and_save_centroids_2d(new_session.id)
        new_session.status = 'SUCCESS'
        new_session.result_message = f'Рекластеризация завершена. Найдено {n_clusters_found} кластеров.'
        original_session.status = 'RECLUSTERED'
        db.session.commit()
        logger.info(f"Синхронная рекластеризация {new_session_id} успешно завершена.")
        return new_session.id
    except Exception as e:
        db.session.rollback()
        logger.error(f"Ошибка в синхронной рекластеризации {new_session_id}: {e}", exc_info=True)
        try:
            fail_session = db.session.get(ClusteringSession, new_session_id)
            if fail_session:
                fail_session.status = 'FAILURE'
                fail_session.result_message = f"Ошибка рекластеризации: {str(e)[:500]}"
            if original_session:
                 original_session.status = 'RECLUSTERING_FAILED'
            db.session.commit()
        except Exception as db_err:
             db.session.rollback()
             logger.error(f"Не удалось обновить статус сессии {new_session_id} на FAILURE: {db_err}", exc_info=True)
        raise

def generate_and_save_scatter_data(session_id, embeddings, labels):
    logger.info(f"Генерация и сохранение данных Scatter Plot для сессии {session_id}")
    scatter_plot_data = None
    pca_time_sec = None
    scatter_cache_file_path = None

    try:
        if embeddings is None or labels is None or embeddings.shape[0] != labels.shape[0]:
            logger.warning(f"Некорректные входные данные для генерации scatter plot в сессии {session_id}")
            return None, None

        num_points = embeddings.shape[0]
        max_points = config.Config.MAX_SCATTER_PLOT_POINTS
        indices = np.arange(num_points)

        if num_points > max_points:
            logger.warning(f"Session {session_id}: Сэмплирование {num_points} точек до {max_points} для scatter plot")
            indices = np.random.choice(indices, max_points, replace=False)
            embeddings_sampled = embeddings[indices]
            labels_sampled = labels[indices]
        else:
            embeddings_sampled = embeddings
            labels_sampled = labels

        if embeddings_sampled.shape[0] < 2:
            logger.warning(f"Недостаточно точек ({embeddings_sampled.shape[0]}) для PCA в сессии {session_id}")
            return None, None

        pca_start_time = time.time()
        pca = PCA(n_components=2, svd_solver='full', random_state=42)
        try:
            embeddings_2d = pca.fit_transform(embeddings_sampled)
            pca_end_time = time.time()
            pca_time_sec = pca_end_time - pca_start_time

            formatted_scatter_data = []
            for i in range(embeddings_2d.shape[0]):
                formatted_scatter_data.append({
                    'x': float(embeddings_2d[i, 0]),
                    'y': float(embeddings_2d[i, 1]),
                    'cluster': str(labels_sampled[i])
                })
            scatter_plot_data = formatted_scatter_data
            logger.info(f"Успешно сгенерированы данные Scatter Plot для сессии {session_id} ({len(scatter_plot_data)} точек) за {pca_time_sec:.2f} сек.")

            scatter_filename = f"scatter_{session_id}.json"
            scatter_folder = current_app.config['SCATTER_DATA_FOLDER']
            scatter_cache_file_path = os.path.join(scatter_folder, scatter_filename)
            cache_content = {
                "scatter_plot_data": scatter_plot_data,
                "pca_time": pca_time_sec
            }
            with open(scatter_cache_file_path, 'w') as f:
                json.dump(cache_content, f)
            logger.info(f"Данные Scatter Plot сохранены в кэш для сессии {session_id}: {scatter_cache_file_path}")

        except ValueError as pca_err:
            logger.error(f"PCA не удался для scatter plot в сессии {session_id}: {pca_err}", exc_info=True)
            scatter_plot_data = {"error": f"Ошибка расчета PCA: {pca_err}"}
            scatter_cache_file_path = None
            pca_time_sec = None
        except (OSError, IOError) as e:
            logger.error(f"Не удалось сохранить кэш Scatter Plot для сессии {session_id}: {e}", exc_info=True)
            scatter_cache_file_path = None

    except Exception as e:
        logger.error(f"Непредвиденная ошибка при генерации/сохранении scatter plot для сессии {session_id}: {e}", exc_info=True)
        scatter_plot_data = {"error": "Внутренняя ошибка генерации scatter plot."}
        scatter_cache_file_path = None
        pca_time_sec = None

    return scatter_cache_file_path, pca_time_sec


def redistribute_cluster_data(session_id, cluster_label_to_remove_str, user_id):
    logger.info(f"Запуск перераспределения для кластера {cluster_label_to_remove_str} в сессии {session_id}")
    session = db.session.get(ClusteringSession, session_id)

    if not session or session.user_id != user_id:
        raise ValueError("Сессия не найдена или доступ запрещен")
    if session.status not in ['SUCCESS', 'RECLUSTERED']:
        raise ValueError(f"Невозможно изменить сессию со статусом {session.status}")

    cluster_to_remove = session.clusters.filter_by(cluster_label=cluster_label_to_remove_str, is_deleted=False).first()
    if not cluster_to_remove:
        raise ValueError(f"Кластер {cluster_label_to_remove_str} не найден или уже удален")

    removed_cluster_display_name = cluster_to_remove.name or f"Кластер {cluster_label_to_remove_str}"

    remaining_clusters = session.clusters.filter(
        ClusterMetadata.cluster_label != cluster_label_to_remove_str,
        ClusterMetadata.is_deleted == False
    ).all()

    old_scatter_cache_path = session.scatter_data_file_path
    if old_scatter_cache_path and os.path.exists(old_scatter_cache_path):
        try:
            os.remove(old_scatter_cache_path)
            logger.info(f"Предварительно удален старый кэш scatter plot: {old_scatter_cache_path}")
        except OSError as e:
            logger.error(f"Ошибка предварительного удаления кэша scatter plot {old_scatter_cache_path}: {e}")
    session.scatter_data_file_path = None

    if not remaining_clusters:
        logger.warning(f"Нет активных кластеров для перераспределения из {cluster_label_to_remove_str}. Кластер будет просто удален.")
        cluster_to_remove.is_deleted = True
        sheet_path = cluster_to_remove.contact_sheet_path
        cluster_to_remove.contact_sheet_path = None
        log_manual_adjustment(session.id, user_id, "DELETE_CLUSTER_NO_TARGETS", {"cluster_label": cluster_label_to_remove_str, "cluster_name": cluster_to_remove.name})

        session.result_message = f"'{removed_cluster_display_name}' удален. Других кластеров для перераспределения не найдено."
        flag_modified(session, "result_message")

        db.session.commit()
        if sheet_path and os.path.exists(sheet_path):
            try: os.remove(sheet_path)
            except OSError as e: logger.error(f"Error deleting contact sheet file {sheet_path}: {e}")
        calculate_and_save_centroids_2d(session.id)
        return {"message": session.result_message}

    try:
        embeddings, image_ids, image_paths = load_embeddings(session.input_file_path)
        if embeddings is None:
            raise ValueError("Не удалось загрузить эмбеддинги для перераспределения.")

        initial_labels = get_cluster_labels_for_session(session, embeddings)
        if initial_labels is None:
             raise RuntimeError(f"Не удалось получить/пересчитать исходные метки для сессии {session_id}.")
        if initial_labels.shape[0] != embeddings.shape[0]:
             raise RuntimeError(f"Размер эмбеддингов ({embeddings.shape[0]}) не совпадает с размером исходных меток ({initial_labels.shape[0]})")

        cluster_label_to_remove_int = int(cluster_label_to_remove_str)

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logger.error(f"Ошибка подготовки данных для перераспределения {session_id}/{cluster_label_to_remove_str}: {e}", exc_info=True)
        raise ValueError(f"Ошибка подготовки данных: {e}") from e

    try:
        point_indices_to_move = np.where(initial_labels == cluster_label_to_remove_int)[0]
        final_labels = np.copy(initial_labels)

        if len(point_indices_to_move) == 0:
            logger.warning(f"Не найдено точек для перераспределения из кластера {cluster_label_to_remove_str}. Кластер будет помечен как удаленный.")
            cluster_to_remove.is_deleted = True
            sheet_path = cluster_to_remove.contact_sheet_path
            cluster_to_remove.contact_sheet_path = None
            log_manual_adjustment(session.id, user_id, "DELETE_CLUSTER_NO_POINTS", {"cluster_label": cluster_label_to_remove_str, "cluster_name": cluster_to_remove.name})

            session.result_message = f"'{removed_cluster_display_name}' удален. Точек для перераспределения не найдено."
            flag_modified(session, "result_message")

            db.session.commit()
            if sheet_path and os.path.exists(sheet_path):
                try: os.remove(sheet_path)
                except OSError as e: logger.error(f"Error deleting contact sheet file {sheet_path}: {e}")
            calculate_and_save_centroids_2d(session.id)

            final_labels[final_labels == cluster_label_to_remove_int] = -1

            new_cache_path, _ = generate_and_save_scatter_data(session.id, embeddings, final_labels)
            if new_cache_path:
                session_reloaded = db.session.get(ClusteringSession, session_id)
                if session_reloaded:
                    session_reloaded.scatter_data_file_path = new_cache_path
                    flag_modified(session_reloaded, "scatter_data_file_path")
                    db.session.commit()
            return {"message": session.result_message}

        embeddings_to_move = embeddings[point_indices_to_move]

        target_centroids = []
        target_cluster_map = {}
        valid_target_clusters = []
        target_label_map = {}

        for i, cluster_meta in enumerate(remaining_clusters):
             centroid_vec = cluster_meta.get_centroid()
             if centroid_vec is not None and centroid_vec.shape[0] == embeddings.shape[1]:
                 target_centroids.append(centroid_vec)
                 db_id = cluster_meta.id
                 target_cluster_map[len(target_centroids) - 1] = db_id
                 valid_target_clusters.append(cluster_meta)
                 target_label_map[db_id] = cluster_meta.cluster_label
             else:
                 logger.warning(f"Пропуск целевого кластера {cluster_meta.id} из-за некорректного или отсутствующего центроида.")

        if not target_centroids:
             logger.error(f"Не найдено валидных центроидов среди оставшихся кластеров для сессии {session_id}. Перераспределение невозможно.")
             raise ValueError("Нет валидных целевых кластеров для перераспределения.")

        target_centroids_np = np.array(target_centroids)
        distances = euclidean_distances(embeddings_to_move, target_centroids_np)
        nearest_target_indices = np.argmin(distances, axis=1)

        redistribution_counts = {}
        for i, point_global_idx in enumerate(point_indices_to_move):
            target_centroid_idx = nearest_target_indices[i]
            target_cluster_db_id = target_cluster_map[target_centroid_idx]
            redistribution_counts[target_cluster_db_id] = redistribution_counts.get(target_cluster_db_id, 0) + 1
            new_label_for_point = int(target_label_map[target_cluster_db_id])
            final_labels[point_global_idx] = new_label_for_point

        cluster_to_remove.is_deleted = True
        sheet_path = cluster_to_remove.contact_sheet_path
        cluster_to_remove.contact_sheet_path = None

        redistribution_log_details = {
            "cluster_label_removed": cluster_label_to_remove_str,
            "cluster_name_removed": cluster_to_remove.name,
            "points_moved": len(point_indices_to_move),
            "targets": []
        }

        for target_db_id, count in redistribution_counts.items():
            target_cluster = next((c for c in valid_target_clusters if c.id == target_db_id), None)
            if target_cluster:
                original_size = target_cluster.size if target_cluster.size else 0
                target_cluster.size = original_size + count
                flag_modified(target_cluster, "size")
                redistribution_log_details["targets"].append({
                    "target_cluster_label": target_cluster.cluster_label,
                    "target_cluster_name": target_cluster.name,
                    "count": count,
                    "new_size": target_cluster.size
                })
            else:
                 logger.error(f"Не удалось найти целевой кластер с ID {target_db_id} для обновления размера.")


        session.num_clusters = len(valid_target_clusters)
        session.result_message = f"'{removed_cluster_display_name}' удален, точки ({len(point_indices_to_move)}) перераспределены."
        flag_modified(session, "num_clusters")
        flag_modified(session, "result_message")

        log_manual_adjustment(session.id, user_id, "REDISTRIBUTE_CLUSTER", redistribution_log_details)

        deleted_clusters_in_db = ClusterMetadata.query.with_session(db.session).filter_by(session_id=session_id, is_deleted=True).all()
        deleted_labels_int = set()
        for dc in deleted_clusters_in_db:
            if dc.id != cluster_to_remove.id:
                 try:
                     deleted_labels_int.add(int(dc.cluster_label))
                 except ValueError:
                     logger.warning(f"Could not parse already deleted cluster label {dc.cluster_label} to int.")

        try:
            deleted_labels_int.add(cluster_label_to_remove_int)
        except ValueError:
            pass

        logger.info(f"Обеспечение установки метки -1 для удаленных кластеров {deleted_labels_int} в final_labels.")
        for del_label in deleted_labels_int:
             mask = (final_labels == del_label)
             if np.any(mask):
                 logger.warning(f"Найдены точки ({np.sum(mask)}) с меткой {del_label} (удаленный кластер) в final_labels. Установка в -1.")
                 final_labels[mask] = -1

        logger.info(f"Генерация scatter plot с перераспределенными и очищенными метками для сессии {session_id}...")
        new_cache_path, _ = generate_and_save_scatter_data(session.id, embeddings, final_labels)
        if new_cache_path:
            session.scatter_data_file_path = new_cache_path
            flag_modified(session, "scatter_data_file_path")
        else:
             logger.error(f"Не удалось сгенерировать или сохранить новый кэш scatter plot для сессии {session_id} после перераспределения.")

        db.session.commit()

        logger.info(f"Данные для кластера {cluster_label_to_remove_str} перераспределены. Обновление 2D координат...")
        calculate_and_save_centroids_2d(session.id)

        if sheet_path and os.path.exists(sheet_path):
            try:
                os.remove(sheet_path)
                logger.info(f"Удален файл контактного отпечатка: {sheet_path}")
            except OSError as e:
                 logger.error(f"Ошибка удаления файла контактного отпечатка {sheet_path}: {e}")

        logger.info(f"Перераспределение для кластера {cluster_label_to_remove_str} в сессии {session_id} завершено успешно.")
        return {"message": session.result_message}

    except Exception as e:
        db.session.rollback()
        logger.error(f"Ошибка выполнения перераспределения {session_id}/{cluster_label_to_remove_str}: {e}", exc_info=True)
        raise RuntimeError(f"Ошибка перераспределения: {e}") from e

def log_manual_adjustment(session_id, user_id, action, details):
    try:
        log_entry = ManualAdjustmentLog(
            session_id=session_id, user_id=user_id, action_type=action
        )
        log_entry.set_details(details)
        db.session.add(log_entry)
        logger.info(f"Logged manual adjustment: Session={session_id}, User={user_id}, Action={action}")
        return True
    except Exception as e:
        logger.error(f"Failed to log manual adjustment for session {session_id}: {e}", exc_info=True)
        return False