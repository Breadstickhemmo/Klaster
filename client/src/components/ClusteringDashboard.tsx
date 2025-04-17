import React, { useState, useCallback, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import SessionSelector from './SessionSelector';
import ClusteringControls from './ClusteringControls';
import SessionDetailsDisplay from './SessionDetailsDisplay';
import ChartsDisplay from './ChartsDisplay';
import ContactSheetsGrid from './ContactSheetsGrid';
import ConfirmationModal from './ConfirmationModal';
import {
    startClustering,
    getClusteringSessions,
    getClusteringResults,
    deleteAndRedistributeCluster,
    renameCluster,
    SessionResultResponse,
    SessionListItem,
    StartClusteringPayload
} from '../services/api';
import '../styles/ClusteringDashboard.css';

type FetchWithAuth = (url: string, options?: RequestInit) => Promise<Response>;
interface ClusteringDashboardProps {
  fetchWithAuth: FetchWithAuth;
}

interface ConfirmModalArgs {
    clusterId: string | number;
    clusterDisplayName: string;
}

const FINAL_STATUSES = ['SUCCESS', 'FAILURE', 'RECLUSTERED', 'RECLUSTERING_FAILED'];
const POLLING_INTERVAL_MS = 5000;

const ClusteringDashboard: React.FC<ClusteringDashboardProps> = ({ fetchWithAuth }) => {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentSessionDetails, setCurrentSessionDetails] = useState<SessionResultResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isFetchingSessions, setIsFetchingSessions] = useState<boolean>(true);
  const [isFetchingResults, setIsFetchingResults] = useState<boolean>(false);
  const [isDeletingId, setIsDeletingId] = useState<string | number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRenamingId, setIsRenamingId] = useState<string | number | null>(null);

  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false);
  const [confirmModalArgs, setConfirmModalArgs] = useState<ConfirmModalArgs | null>(null);
  const [isConfirmLoading, setIsConfirmLoading] = useState(false);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
      console.log("Polling stopped.");
    }
  }, []);

  const fetchSessionResults = useCallback(async (sessionId: string, isPolling = false) => {
    if (!isPolling) {
       setIsFetchingResults(true);
       setError(null);
    }
    console.log(`${isPolling ? 'Polling' : 'Fetching'} results for session ${sessionId}...`);

    try {
      const resultsData = await getClusteringResults(fetchWithAuth, sessionId);

      setCurrentSessionDetails(resultsData);

      setSessions(prevSessions =>
          prevSessions.map(s =>
              s.session_id === sessionId && (s.status !== resultsData.status || s.result_message !== resultsData.message)
                  ? { ...s, status: resultsData.status, result_message: resultsData.message || s.result_message }
                  : s
          )
      );

      if (FINAL_STATUSES.includes(resultsData.status)) {
        console.log(`Session ${sessionId} reached final status: ${resultsData.status}. Stopping polling.`);
        stopPolling();
      }

      if (resultsData.error && isPolling) {
          console.warn(`Polling for session ${sessionId} received error status: ${resultsData.status} - ${resultsData.error}`);
      }

    } catch (err: any) {
      console.error(`Error ${isPolling ? 'polling' : 'fetching'} results for session ${sessionId}:`, err);
      if (!isPolling) {
          const errorMsg = err.message || `Не удалось загрузить результаты для сессии ${sessionId}.`;
          setError(errorMsg);
          toast.error(errorMsg);
          setCurrentSessionDetails(null);
      }
      if (err.message?.includes('404') || err.message?.includes('401')) {
           stopPolling();
      }
    } finally {
      if (!isPolling) {
         setIsFetchingResults(false);
      }
    }
  }, [fetchWithAuth, stopPolling]);

  const startPolling = useCallback((sessionId: string) => {
    stopPolling();
    console.log(`Starting polling for session ${sessionId}...`);
    fetchSessionResults(sessionId, true);
    pollingIntervalRef.current = setInterval(() => {
      fetchSessionResults(sessionId, true);
    }, POLLING_INTERVAL_MS);
  }, [stopPolling, fetchSessionResults]);

  useEffect(() => {
    const fetchInitialSessions = async () => {
      setIsFetchingSessions(true);
      setError(null);
      try {
        const fetchedSessions = await getClusteringSessions(fetchWithAuth);
        setSessions(fetchedSessions);
      } catch (err: any) {
        console.error("Error fetching sessions:", err);
        const errorMsg = err.message || 'Не удалось загрузить список сессий.';
        setError(errorMsg);
      } finally {
        setIsFetchingSessions(false);
      }
    };
    fetchInitialSessions();
    return () => {
        stopPolling();
    };
  }, [fetchWithAuth, stopPolling]);

  useEffect(() => {
    stopPolling();

    if (currentSessionId) {
      fetchSessionResults(currentSessionId).then(() => {
         setTimeout(() => {
             setCurrentSessionDetails(prevDetails => {
                if (prevDetails && prevDetails.session_id === currentSessionId && !FINAL_STATUSES.includes(prevDetails.status)) {
                     startPolling(currentSessionId);
                }
                return prevDetails;
             });
         }, 0);
      });
    } else {
      setCurrentSessionDetails(null);
      setError(null);
    }
  }, [currentSessionId, fetchSessionResults, startPolling, stopPolling]);

  const handleSelectSession = useCallback((sessionId: string) => {
      if (sessionId !== currentSessionId) {
          setCurrentSessionId(sessionId);
      }
  }, [currentSessionId]);

  const handleStartClustering = useCallback(async (payload: StartClusteringPayload) => {
    setIsLoading(true);
    setError(null);
    stopPolling();
    toast.info(`Запускаем кластеризацию (${payload.algorithm.toUpperCase()})...`);
    try {
        const response = await startClustering(fetchWithAuth, payload);
        toast.success(`Кластеризация запущена! ID сессии: ${response.session_id}`);
        const newSessionItem: SessionListItem = {
            session_id: response.session_id, created_at: new Date().toISOString(), status: 'STARTED',
            algorithm: payload.algorithm, params: payload.params, num_clusters: null,
            result_message: "Запущено...", original_filename: payload.embeddingFile.name
        };
        setSessions(prev => [newSessionItem, ...prev]);
        setCurrentSessionId(response.session_id);
    } catch (err: any) {
        console.error("Clustering start error:", err);
        const errorMsg = err.message || 'Не удалось запустить кластеризацию.';
        setError(errorMsg);
        toast.error(`Ошибка запуска кластеризации: ${errorMsg}`);
        throw err;
    } finally {
        setIsLoading(false);
    }
  }, [fetchWithAuth, stopPolling]);

   const handleDeleteAndRedistributeCluster = useCallback(async (clusterLabel: string | number) => {
    if (!currentSessionId || !currentSessionDetails) { toast.error("Нет активной сессии или данных для выполнения операции."); return; }

    const labelToDelete = String(clusterLabel);
    const clusterToDelete = currentSessionDetails.clusters.find(c => String(c.id) === labelToDelete);
    const clusterDisplayName = clusterToDelete?.name || `Кластер ${labelToDelete}`;

    setConfirmModalArgs({ clusterId: clusterLabel, clusterDisplayName });
    setIsConfirmModalOpen(true);

  }, [currentSessionId, currentSessionDetails]);

  const confirmDeletion = useCallback(async () => {
      if (!currentSessionId || !confirmModalArgs) return;

      const { clusterId, clusterDisplayName } = confirmModalArgs;

      setIsConfirmLoading(true);
      setIsDeletingId(clusterId);
      setError(null);
      stopPolling();

      toast.info(`Удаляем '${clusterDisplayName}' и перераспределяем его точки...`);

      try {
          const response = await deleteAndRedistributeCluster(fetchWithAuth, currentSessionId, String(clusterId));
          toast.success(response.message || `'${clusterDisplayName}' удален, точки перераспределены.`);
          await fetchSessionResults(currentSessionId);
          setSessions(prevSessions =>
              prevSessions.map(s =>
                  s.session_id === currentSessionId
                      ? { ...s, result_message: response.message || s.result_message }
                      : s
              )
          );

          setCurrentSessionDetails(prevDetails => {
               if (prevDetails && prevDetails.session_id === currentSessionId && !FINAL_STATUSES.includes(prevDetails.status)) {
                    startPolling(currentSessionId);
               }
               return prevDetails;
           });

      } catch (err: any) {
          console.error(`Error deleting/redistributing cluster ${clusterId}:`, err);
          const errorMsg = err.message || `Не удалось удалить/перераспределить '${clusterDisplayName}'.`;
          setError(errorMsg);
          toast.error(errorMsg);
          fetchSessionResults(currentSessionId);
      } finally {
          setIsConfirmLoading(false);
          setIsDeletingId(null);
          setIsConfirmModalOpen(false);
          setConfirmModalArgs(null);
      }
  }, [currentSessionId, confirmModalArgs, fetchWithAuth, stopPolling, fetchSessionResults, startPolling]);

  const closeConfirmModal = () => {
       setIsConfirmModalOpen(false);
       setConfirmModalArgs(null);
   };

  const handleRenameCluster = useCallback(async (clusterId: string | number, newName: string): Promise<boolean> => {
    if (!currentSessionId) {
        toast.error("Нет активной сессии для переименования.");
        return false;
    }
    const clusterToRename = currentSessionDetails?.clusters.find(c => String(c.id) === String(clusterId));
    const oldDisplayName = clusterToRename?.name || `Кластер ${clusterId}`;

    setIsRenamingId(clusterId);
    let success = false;

    try {
        await renameCluster(fetchWithAuth, currentSessionId, clusterId, newName);
        await fetchSessionResults(currentSessionId);
        success = true;

    } catch (err: any) {
        console.error(`Error renaming cluster ${clusterId}:`, err);
        const errorMsg = err.message || `Не удалось переименовать '${oldDisplayName}'.`;
        setError(errorMsg);
        toast.error(errorMsg);
        success = false;
    } finally {
        setIsRenamingId(null);
    }
    return success;
  }, [currentSessionId, currentSessionDetails, fetchWithAuth, fetchSessionResults]);

  const isProcessing = isLoading || isConfirmLoading || isFetchingResults || isFetchingSessions || isRenamingId !== null || isDeletingId !== null;
  const isCurrentSessionLoading = isFetchingResults && !pollingIntervalRef.current;

  return (
    <div className="clustering-dashboard">
      <h2>Панель управления кластеризацией</h2>

      <SessionSelector
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        disabled={isProcessing}
        isLoading={isFetchingSessions}
        error={error && !currentSessionId ? error : null}
      />

      <ClusteringControls
        onStartClustering={handleStartClustering}
        disabled={isProcessing}
      />

      {currentSessionId && isCurrentSessionLoading && (
        <div className="card status-card"> <p>Загрузка результатов для сессии {currentSessionId.substring(0,8)}...</p> </div>
      )}

      {currentSessionId && !isCurrentSessionLoading && !currentSessionDetails && error && (
           <div className="card status-card error-message"><p>Ошибка загрузки результатов: {error}</p></div>
       )}

      {currentSessionId && currentSessionDetails && (
          <>
              <SessionDetailsDisplay details={currentSessionDetails} />

              {(currentSessionDetails.status === 'SUCCESS' || currentSessionDetails.status === 'RECLUSTERED' || currentSessionDetails.status === 'PROCESSING') && (
                  <ChartsDisplay
                      details={currentSessionDetails}
                      sessionId={currentSessionId}
                  />
              )}

               {(currentSessionDetails.status === 'SUCCESS' || currentSessionDetails.status === 'RECLUSTERED') && (
                  <ContactSheetsGrid
                    clusters={currentSessionDetails.clusters || []}
                    sessionId={currentSessionId}
                    onRedistribute={handleDeleteAndRedistributeCluster}
                    onRename={handleRenameCluster}
                    isDeletingId={isDeletingId}
                    disabled={isProcessing || !['SUCCESS', 'RECLUSTERED'].includes(currentSessionDetails.status)}
                    status={currentSessionDetails.status}
                  />
              )}

              {currentSessionDetails.status !== 'SUCCESS' && currentSessionDetails.status !== 'RECLUSTERED' && currentSessionDetails.status !== 'PROCESSING' && (
                 <div className="card status-card">
                     <p>Статус сессии {currentSessionId.substring(0,8)}...: <strong>{currentSessionDetails.status}</strong>.</p>
                     {currentSessionDetails.message && <p>{currentSessionDetails.message}</p>}
                     {currentSessionDetails.error && <p className="error-message">{currentSessionDetails.error}</p>}
                     <p>Результаты не могут быть отображены или обработаны в данный момент.</p>
                 </div>
              )}
              {currentSessionDetails.status === 'PROCESSING' && !isCurrentSessionLoading && (
                   <div className="card status-card">
                       <p>Сессия {currentSessionId.substring(0,8)}... обрабатывается. Статус будет обновлен автоматически.</p>
                       {currentSessionDetails.message && <p>{currentSessionDetails.message}</p>}
                   </div>
              )}
          </>
       )}

        {!currentSessionId && !isProcessing && sessions.length > 0 && ( <div className="card status-card"> <p>Выберите сессию из списка выше для просмотра результатов или запустите новую кластеризацию.</p> </div> )}
        {!currentSessionId && !isProcessing && sessions.length === 0 && !isFetchingSessions && !error && ( <div className="card status-card"> <p>Нет доступных сессий. Запустите новую кластеризацию, используя форму выше.</p> </div> )}

        <ConfirmationModal
            isOpen={isConfirmModalOpen}
            onClose={closeConfirmModal}
            onConfirm={confirmDeletion}
            title="Подтверждение удаления"
            message={
                confirmModalArgs ? (
                    <>
                        Вы уверены, что хотите удалить <br />
                        <strong>'{confirmModalArgs.clusterDisplayName}'</strong>?
                        <br />
                        Его точки будут перераспределены по ближайшим оставшимся кластерам.
                        Это действие необратимо для текущей сессии.
                    </>
                ) : (
                    'Подтвердите действие.'
                )
            }
            confirmText="Удалить и перераспределить"
            cancelText="Отмена"
            confirmButtonClass="danger-btn"
            isLoading={isConfirmLoading}
        />

    </div>
  );
};

export default ClusteringDashboard;