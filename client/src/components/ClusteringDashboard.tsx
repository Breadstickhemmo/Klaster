import React, { useState, useCallback, useEffect } from 'react';
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

const ClusteringDashboard: React.FC<ClusteringDashboardProps> = ({ fetchWithAuth }) => {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentSessionDetails, setCurrentSessionDetails] = useState<SessionResultResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isFetchingSessions, setIsFetchingSessions] = useState<boolean>(true);
  const [isFetchingResults, setIsFetchingResults] = useState<boolean>(false);
  const [isDeletingId, setIsDeletingId] = useState<string | number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSessions = async () => {
      setIsFetchingSessions(true);
      setError(null);
      try {
        const fetchedSessions = await getClusteringSessions(fetchWithAuth);
        setSessions(fetchedSessions);
        if (!currentSessionId && fetchedSessions.length > 0) {
            setCurrentSessionId(fetchedSessions[0].session_id);
        }
      } catch (err: any) {
        console.error("Error fetching sessions:", err);
        const errorMsg = err.message || 'Не удалось загрузить список сессий.';
        setError(errorMsg);
      } finally {
        setIsFetchingSessions(false);
      }
    };
    fetchSessions();
  }, [fetchWithAuth]);

  useEffect(() => {
    if (!currentSessionId) {
      setCurrentSessionDetails(null);
      return;
    }
    const fetchResults = async () => {
      setIsFetchingResults(true);
      setError(null);
      setCurrentSessionDetails(null);
      try {
        const resultsData = await getClusteringResults(fetchWithAuth, currentSessionId);
        setCurrentSessionDetails(resultsData);
        if (resultsData.status !== 'SUCCESS' && resultsData.status !== 'RECLUSTERED') {
            toast.info(`Статус сессии ${resultsData.session_id.substring(0,8)}...: ${resultsData.status}. ${resultsData.message || resultsData.error || ''}`);
        } else if (!resultsData.clusters || resultsData.clusters.length === 0) {
             toast.info(`Сессия ${resultsData.session_id.substring(0,8)}... (${resultsData.algorithm}) завершена, но кластеры не найдены.`);
        }
      } catch (err: any) {
        console.error(`Error fetching results for session ${currentSessionId}:`, err);
        const errorMsg = err.message || `Не удалось загрузить результаты для сессии ${currentSessionId}.`;
        setError(errorMsg);
        toast.error(errorMsg);
        setCurrentSessionDetails(null);
      } finally {
        setIsFetchingResults(false);
      }
    };
    fetchResults();
  }, [currentSessionId, fetchWithAuth]);

  const handleSelectSession = useCallback((sessionId: string) => {
      setCurrentSessionId(sessionId);
      setError(null);
  }, []);

  const handleStartClustering = useCallback(async (payload: StartClusteringPayload) => {
    setIsLoading(true);
    setError(null);
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
  }, [fetchWithAuth]);

   const handleDeleteClusterAndRecluster = useCallback(async (clusterLabel: string | number) => {
    if (!currentSessionId) { toast.error("Нет активной сессии для выполнения операции."); return; }
    const labelToDelete = String(clusterLabel);
    setIsDeletingId(clusterLabel); setIsLoading(true); setError(null);
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
  }, [currentSessionId, fetchWithAuth, sessions]);

  const isProcessing = isLoading || isFetchingSessions || isFetchingResults || isDeletingId !== null;

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

      {isFetchingResults && currentSessionId && (
        <div className="card status-card"> <p>Загрузка результатов для сессии {currentSessionId}...</p> </div>
      )}

      {currentSessionId && !isFetchingResults && !currentSessionDetails && error && (
           <div className="card status-card error-message"><p>Ошибка загрузки результатов: {error}</p></div>
       )}

      {currentSessionId && !isFetchingResults && currentSessionDetails && (
          <>
              <SessionDetailsDisplay details={currentSessionDetails} />

              <ChartsDisplay
                  details={currentSessionDetails}
                  sessionId={currentSessionId}
              />

              <ContactSheetsGrid
                clusters={currentSessionDetails.clusters || []}
                sessionId={currentSessionId}
                onDelete={handleDeleteClusterAndRecluster}
                isDeletingId={isDeletingId}
                disabled={isProcessing}
                status={currentSessionDetails.status}
              />

              {currentSessionDetails.status !== 'SUCCESS' && currentSessionDetails.status !== 'RECLUSTERED' && currentSessionDetails.status !== 'PROCESSING' && (
                 <div className="card status-card">
                     <p>Статус сессии {currentSessionId}: {currentSessionDetails.status}.</p>
                     {currentSessionDetails.message && <p>{currentSessionDetails.message}</p>}
                     {currentSessionDetails.error && <p className="error-message">{currentSessionDetails.error}</p>}
                     <p>Результаты не могут быть отображены.</p>
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