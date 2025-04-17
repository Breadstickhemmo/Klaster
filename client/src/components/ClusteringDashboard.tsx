import React, { useState, useCallback, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import SessionSelector from './SessionSelector';
import ClusteringControls from './ClusteringControls';
import SessionDetailsDisplay from './SessionDetailsDisplay';
import ChartsDisplay from './ChartsDisplay';
import ContactSheetsGrid from './ContactSheetsGrid';
import {
    startClustering,
    getClusteringSessions,
    getClusteringResults,
    deleteClusterAndRecluster,
    SessionResultResponse,
    SessionListItem,
    StartClusteringPayload
} from '../services/api';
import '../styles/ClusteringDashboard.css';

type FetchWithAuth = (url: string, options?: RequestInit) => Promise<Response>;
interface ClusteringDashboardProps {
  fetchWithAuth: FetchWithAuth;
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
              s.session_id === sessionId && s.status !== resultsData.status
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

   const handleDeleteClusterAndRecluster = useCallback(async (clusterLabel: string | number) => {
    if (!currentSessionId) { toast.error("Нет активной сессии для выполнения операции."); return; }
    const labelToDelete = String(clusterLabel);
    setIsDeletingId(clusterLabel); setIsLoading(true); setError(null); stopPolling();
    toast.info(`Удаляем кластер ${labelToDelete} и запускаем рекластеризацию...`);
    try {
        const response = await deleteClusterAndRecluster(fetchWithAuth, currentSessionId, labelToDelete);
        toast.success(`Кластер ${labelToDelete} удален. Создана новая сессия: ${response.new_session_id}`);
        const originalSession = sessions.find(s => s.session_id === currentSessionId);
        const newSessionItem: SessionListItem = {
            session_id: response.new_session_id, created_at: new Date().toISOString(), status: 'STARTED',
            algorithm: originalSession?.algorithm || '', params: originalSession?.params || {}, num_clusters: null,
            result_message: "Рекластеризация...", original_filename: originalSession?.original_filename || null
        };
        setSessions(prev => [newSessionItem, ...prev.map(s => s.session_id === currentSessionId ? { ...s, status: 'RECLUSTERED' } : s )]);
        setCurrentSessionId(response.new_session_id);
    } catch (err: any) {
        console.error(`Error deleting/re-clustering cluster ${labelToDelete}:`, err);
        const errorMsg = err.message || `Не удалось удалить кластер ${labelToDelete}.`;
        setError(errorMsg); toast.error(errorMsg);
    } finally { setIsDeletingId(null); setIsLoading(false); }
  }, [currentSessionId, fetchWithAuth, sessions, stopPolling]);

  const isProcessing = isLoading || isDeletingId !== null;
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
                    onDelete={handleDeleteClusterAndRecluster}
                    isDeletingId={isDeletingId}
                    disabled={isProcessing || !FINAL_STATUSES.includes(currentSessionDetails.status)}
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

    </div>
  );
};

export default ClusteringDashboard;