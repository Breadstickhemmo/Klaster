import os
import logging
import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sklearn.metrics import euclidean_distances
from sqlalchemy.orm.attributes import flag_modified
from models import db, ClusteringSession, ClusterMetadata, ManualAdjustmentLog
from algorithms import get_cluster_labels_for_session
from data_loader import load_embeddings
from visualization import calculate_and_save_centroids_2d, generate_and_save_scatter_data

logger = logging.getLogger(__name__)

def log_manual_adjustment(session_id, user_id, action, details):
    try:
        log_entry = ManualAdjustmentLog(
            session_id=session_id,
            user_id=user_id,
            action_type=action
        )
        log_entry.set_details(details)
        db.session.add(log_entry)
        db.session.commit()
        logger.info(f"Залогировано ручное изменение: Сессия={session_id}, Пользователь={user_id}, Действие={action}")
        return True
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Не удалось залогировать ручное изменение для сессии {session_id}: {e}", exc_info=True)
        return False
    except Exception as e:
         db.session.rollback()
         logger.error(f"Неожиданная ошибка при логировании ручного изменения для сессии {session_id}: {e}", exc_info=True)
         return False

def redistribute_cluster_data(session_id, cluster_label_to_remove_str, user_id):
    logger.info(f"Начало перераспределения для кластера {cluster_label_to_remove_str} в сессии {session_id}")
    session = db.session.get(ClusteringSession, session_id)

    if not session:
        raise ValueError("Сессия не найдена")
    if session.user_id != user_id:
         raise ValueError("Доступ к сессии запрещен")
    if session.status not in ['SUCCESS', 'RECLUSTERED']:
        raise ValueError(f"Невозможно изменить сессию со статусом {session.status}")

    cluster_to_remove = session.clusters.filter_by(
        cluster_label=cluster_label_to_remove_str,
        is_deleted=False
    ).first()

    if not cluster_to_remove:
        logger.warning(f"Кластер {cluster_label_to_remove_str} не найден или уже удален в сессии {session_id}.")
        raise ValueError(f"Кластер {cluster_label_to_remove_str} не найден или уже удален")

    sheet_path_to_delete = cluster_to_remove.contact_sheet_path
    removed_cluster_display_name = cluster_to_remove.name or f"Кластер {cluster_label_to_remove_str}"
    db_cluster_id_to_remove = cluster_to_remove.id

    remaining_clusters = session.clusters.filter(
        ClusterMetadata.id != db_cluster_id_to_remove,
        ClusterMetadata.is_deleted == False
    ).all()

    old_scatter_cache_path = session.scatter_data_file_path
    if old_scatter_cache_path and os.path.exists(old_scatter_cache_path):
        try:
            os.remove(old_scatter_cache_path)
            logger.info(f"Удален старый кэш scatter plot: {old_scatter_cache_path}")
        except OSError as e:
            logger.error(f"Ошибка удаления старого кэша scatter plot {old_scatter_cache_path}: {e}")
    session.scatter_data_file_path = None
    flag_modified(session, "scatter_data_file_path")
    db.session.commit()

    if not remaining_clusters:
        logger.warning(f"Нет активных кластеров для перераспределения из кластера {cluster_label_to_remove_str}. Кластер будет помечен как удаленный.")
        cluster_to_remove.is_deleted = True
        cluster_to_remove.contact_sheet_path = None
        flag_modified(cluster_to_remove, "is_deleted")
        flag_modified(cluster_to_remove, "contact_sheet_path")

        log_details = {"cluster_label": cluster_label_to_remove_str, "cluster_name": cluster_to_remove.name}
        log_manual_adjustment(session.id, user_id, "DELETE_CLUSTER_NO_TARGETS", log_details)

        session.result_message = f"'{removed_cluster_display_name}' удален. Других кластеров для перераспределения не найдено."
        session.num_clusters = 0
        flag_modified(session, "result_message")
        flag_modified(session, "num_clusters")

        try:
            db.session.commit()
            if sheet_path_to_delete and os.path.exists(sheet_path_to_delete):
                try: os.remove(sheet_path_to_delete)
                except OSError as e: logger.error(f"Ошибка удаления файла контактного листа {sheet_path_to_delete}: {e}")

            calculate_and_save_centroids_2d(session.id)
            try:
                 embeddings, image_ids = load_embeddings(session.input_file_path)
                 if embeddings is not None and image_ids is not None:
                      final_labels_after_delete = np.full(embeddings.shape[0], -1, dtype=int)
                      new_cache_path, _ = generate_and_save_scatter_data(session.id, embeddings, final_labels_after_delete)
                      if new_cache_path:
                           session_reloaded = db.session.get(ClusteringSession, session_id)
                           if session_reloaded:
                               session_reloaded.scatter_data_file_path = new_cache_path
                               flag_modified(session_reloaded, "scatter_data_file_path")
                               db.session.commit()
            except FileNotFoundError:
                 logger.error(f"Не найден файл эмбеддингов {session.input_file_path} для генерации scatter plot после удаления последнего кластера сессии {session.id}")
            except Exception as scatter_err:
                 logger.error(f"Ошибка генерации scatter plot после удаления последнего кластера {session_id}: {scatter_err}")

            return {"message": session.result_message}
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Ошибка БД при удалении последнего кластера {session_id}/{cluster_label_to_remove_str}: {e}", exc_info=True)
            raise RuntimeError("Ошибка БД при удалении последнего кластера") from e

    try:
         embeddings, image_ids = load_embeddings(session.input_file_path)
         if embeddings is None:
             raise ValueError("Не удалось загрузить эмбеддинги для перераспределения.")
         if image_ids is None or len(image_ids) != embeddings.shape[0]:
             logger.warning(f"Отсутствуют или не совпадают ID изображений при перераспределении для сессии {session_id}. Использование индексов.")
             image_ids = [str(i) for i in range(embeddings.shape[0])]

         initial_labels = get_cluster_labels_for_session(session, embeddings)
         if initial_labels is None:
             raise RuntimeError(f"Не удалось получить исходные метки для сессии {session_id}.")
         if initial_labels.shape[0] != embeddings.shape[0]:
             raise RuntimeError(f"Размер эмбеддингов ({embeddings.shape[0]}) не совпадает с размером исходных меток ({initial_labels.shape[0]})")

         try:
            cluster_label_to_remove_int = int(cluster_label_to_remove_str)
         except ValueError:
             raise ValueError(f"Некорректный ID кластера для удаления: {cluster_label_to_remove_str}")

         point_indices_to_move = np.where(initial_labels == cluster_label_to_remove_int)[0]

         if len(point_indices_to_move) == 0:
             logger.warning(f"Не найдено точек с меткой {cluster_label_to_remove_int} для перераспределения в сессии {session_id}. Кластер будет помечен как удаленный.")
             cluster_to_remove.is_deleted = True
             cluster_to_remove.contact_sheet_path = None
             flag_modified(cluster_to_remove, "is_deleted")
             flag_modified(cluster_to_remove, "contact_sheet_path")

             log_details = {"cluster_label": cluster_label_to_remove_str, "cluster_name": cluster_to_remove.name}
             log_manual_adjustment(session.id, user_id, "DELETE_CLUSTER_NO_POINTS", log_details)

             session.result_message = f"'{removed_cluster_display_name}' удален. Точек для перераспределения не найдено."
             flag_modified(session, "result_message")
             db.session.commit()

             if sheet_path_to_delete and os.path.exists(sheet_path_to_delete):
                 try: os.remove(sheet_path_to_delete)
                 except OSError as e: logger.error(f"Ошибка удаления файла контактного листа {sheet_path_to_delete}: {e}")

             calculate_and_save_centroids_2d(session.id)

             final_labels = np.copy(initial_labels)
             final_labels[final_labels == cluster_label_to_remove_int] = -1
             new_cache_path, _ = generate_and_save_scatter_data(session.id, embeddings, final_labels)
             if new_cache_path:
                  session_reloaded = db.session.get(ClusteringSession, session_id)
                  if session_reloaded:
                      session_reloaded.scatter_data_file_path = new_cache_path
                      flag_modified(session_reloaded, "scatter_data_file_path")
                      db.session.commit()

             return {"message": session.result_message}

         logger.info(f"Перераспределение {len(point_indices_to_move)} точек из кластера {cluster_label_to_remove_str}...")
         embeddings_to_move = embeddings[point_indices_to_move]

         target_centroids = []
         target_idx_to_cluster_meta = {}
         for i, cluster_meta in enumerate(remaining_clusters):
             centroid_vec = cluster_meta.get_centroid()
             if centroid_vec is not None and centroid_vec.shape[0] == embeddings.shape[1]:
                 target_centroids.append(centroid_vec)
                 target_idx_to_cluster_meta[len(target_centroids) - 1] = cluster_meta
             else:
                 logger.warning(f"Пропуск оставшегося кластера {cluster_meta.cluster_label} как целевого из-за невалидного центроида.")

         if not target_centroids:
              logger.error(f"Не найдено валидных целевых центроидов для перераспределения в сессии {session_id}.")
              raise ValueError("Нет валидных целевых кластеров для перераспределения.")

         target_centroids_np = np.array(target_centroids)
         distances = euclidean_distances(embeddings_to_move, target_centroids_np)
         nearest_target_indices = np.argmin(distances, axis=1)

         cluster_to_remove.is_deleted = True
         cluster_to_remove.contact_sheet_path = None
         flag_modified(cluster_to_remove, "is_deleted")
         flag_modified(cluster_to_remove, "contact_sheet_path")

         redistribution_counts = {}
         final_labels = np.copy(initial_labels)
         for i, point_global_idx in enumerate(point_indices_to_move):
             target_internal_idx = nearest_target_indices[i]
             target_cluster_meta = target_idx_to_cluster_meta[target_internal_idx]
             target_db_id = target_cluster_meta.id
             target_final_label = int(target_cluster_meta.cluster_label)

             final_labels[point_global_idx] = target_final_label
             redistribution_counts[target_db_id] = redistribution_counts.get(target_db_id, 0) + 1

         redistribution_log_details = {
             "cluster_label_removed": cluster_label_to_remove_str,
             "cluster_name_removed": cluster_to_remove.name,
             "points_moved": len(point_indices_to_move),
             "targets": []
         }
         for target_db_id, count in redistribution_counts.items():
             target_cluster = db.session.get(ClusterMetadata, target_db_id)
             if target_cluster and not target_cluster.is_deleted:
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
                  logger.error(f"Целевой кластер {target_db_id} не найден или удален при обновлении размера.")

         session.num_clusters = len(remaining_clusters)
         session.result_message = f"'{removed_cluster_display_name}' удален, {len(point_indices_to_move)} точек перераспределены."
         session.status = 'RECLUSTERED'
         flag_modified(session, "num_clusters")
         flag_modified(session, "result_message")
         flag_modified(session, "status")

         log_manual_adjustment(session.id, user_id, "REDISTRIBUTE_CLUSTER", redistribution_log_details)

         logger.info(f"Генерация scatter plot с перераспределенными метками для сессии {session_id}...")
         all_deleted_clusters = ClusterMetadata.query.with_session(db.session).filter_by(session_id=session_id, is_deleted=True).all()
         all_deleted_labels_int = set()
         for dc in all_deleted_clusters:
              try: all_deleted_labels_int.add(int(dc.cluster_label))
              except (ValueError, TypeError): pass

         for del_label_int in all_deleted_labels_int:
              mask = (final_labels == del_label_int)
              if np.any(mask):
                  logger.warning(f"Обнаружены точки с удаленной меткой {del_label_int} в final_labels. Установка в -1.")
                  final_labels[mask] = -1

         new_cache_path, _ = generate_and_save_scatter_data(session.id, embeddings, final_labels)
         if new_cache_path:
             session.scatter_data_file_path = new_cache_path
             flag_modified(session, "scatter_data_file_path")
         else:
              logger.error(f"Не удалось сгенерировать/сохранить новый кэш scatter plot для сессии {session_id} после перераспределения.")

         db.session.commit()

         logger.info(f"Перераспределение для кластера {cluster_label_to_remove_str} успешно. Обновление 2D координат центроидов...")
         calculate_and_save_centroids_2d(session.id)

         if sheet_path_to_delete and os.path.exists(sheet_path_to_delete):
             try:
                 os.remove(sheet_path_to_delete)
                 logger.info(f"Удален файл контактного листа для удаленного кластера: {sheet_path_to_delete}")
             except OSError as e:
                  logger.error(f"Ошибка удаления файла контактного листа {sheet_path_to_delete}: {e}")

         logger.info(f"Процесс перераспределения для кластера {cluster_label_to_remove_str} в сессии {session_id} завершен.")
         return {"message": session.result_message}

    except (ValueError, RuntimeError, FileNotFoundError) as e:
         db.session.rollback()
         logger.error(f"Ошибка подготовки/выполнения перераспределения {session_id}/{cluster_label_to_remove_str}: {e}", exc_info=True)
         raise RuntimeError(f"Ошибка подготовки/выполнения перераспределения: {e}") from e
    except SQLAlchemyError as e:
         db.session.rollback()
         logger.error(f"Ошибка БД при перераспределении {session_id}/{cluster_label_to_remove_str}: {e}", exc_info=True)
         raise RuntimeError("Ошибка БД при перераспределении кластера") from e
    except Exception as e:
        db.session.rollback()
        logger.error(f"Неожиданная ошибка при перераспределении {session_id}/{cluster_label_to_remove_str}: {e}", exc_info=True)
        raise RuntimeError(f"Непредвиденная ошибка сервера при перераспределении: {e}") from e