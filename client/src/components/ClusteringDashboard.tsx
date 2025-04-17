import React, { useState, useCallback, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import SessionSelector from './SessionSelector';
import ClusteringControls from './ClusteringControls';
import SessionDetailsDisplay from './SessionDetailsDisplay';
import ChartsDisplay from './ChartsDisplay';
import ContactSheetsGrid from './ContactSheetsGrid';
import ConfirmationModal from './ConfirmationModal';
import SplitClusterModal from './SplitClusterModal'; // Import the new modal
import ExportControls from './ExportControls';
import {
    startClustering,
    getClusteringSessions,
    getClusteringResults,
    deleteAndRedistributeCluster,
    renameCluster,
    mergeSelectedClusters, // Import new API calls
    splitSelectedCluster,  // Import new API calls
    SessionResultResponse,
    SessionListItem,
    StartClusteringPayload
} from '../services/api';
import '../styles/ClusteringDashboard.css';
import '../styles/ContactSheet.css'; // Import for potential merge button styling

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
  const [isLoading, setIsLoading] = useState<boolean>(false); // General loading for start/fetch
  const [isFetchingSessions, setIsFetchingSessions] = useState<boolean>(true);
  const [isFetchingResults, setIsFetchingResults] = useState<boolean>(false);
  const [isDeletingId, setIsDeletingId] = useState<string | number | null>(null); // Specific to delete op
  const [isRenamingId, setIsRenamingId] = useState<string | number | null>(null); // Specific to rename op
  const [isMerging, setIsMerging] = useState<boolean>(false); // Specific to merge op
  const [isSplitting, setIsSplitting] = useState<boolean>(false); // Specific to split op
  const [error, setError] = useState<string | null>(null);

  // State for Delete Confirmation Modal
  const [isConfirmDeleteModalOpen, setIsConfirmDeleteModalOpen] = useState(false);
  const [confirmDeleteArgs, setConfirmDeleteArgs] = useState<ConfirmModalArgs | null>(null);
  const [isConfirmDeleteLoading, setIsConfirmDeleteLoading] = useState(false);

  // State for Merge Selection
  const [selectedClusterIds, setSelectedClusterIds] = useState<Set<string | number>>(new Set());

  // State for Split Modal
  const [splitTarget, setSplitTarget] = useState<{ id: string | number; name: string; } | null>(null);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Combined processing state checker
  const isProcessingAny = isLoading || isFetchingSessions || isFetchingResults || isConfirmDeleteLoading || isDeletingId !== null || isRenamingId !== null || isMerging || isSplitting;

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
              s.session_id === sessionId && (s.status !== resultsData.status || s.result_message !== resultsData.message || s.num_clusters !== resultsData.num_clusters)
                  ? { ...s, status: resultsData.status, result_message: resultsData.message || s.result_message, num_clusters: resultsData.num_clusters }
                  : s
          )
      );
      if (FINAL_STATUSES.includes(resultsData.status)) {
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
          setCurrentSessionDetails(null); // Clear details on fetch error
      }
      if (err.message?.includes('404') || err.message?.includes('401') || err.status === 404) {
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
    // Initial fetch before interval starts
    fetchSessionResults(sessionId, true);
    pollingIntervalRef.current = setInterval(() => {
      fetchSessionResults(sessionId, true);
    }, POLLING_INTERVAL_MS);
  }, [stopPolling, fetchSessionResults]);

  // Fetch initial sessions
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
        // Don't toast here, show error in selector area
      } finally {
        setIsFetchingSessions(false);
      }
    };
    fetchInitialSessions();
    // Cleanup polling on component unmount
    return () => {
        stopPolling();
    };
  }, [fetchWithAuth, stopPolling]); // Only run once on mount


  // Fetch/Poll results when session changes
  useEffect(() => {
    stopPolling(); // Stop any previous polling
    setSelectedClusterIds(new Set()); // Clear selection when session changes
    setSplitTarget(null); // Clear split target

    if (currentSessionId) {
      fetchSessionResults(currentSessionId).then(() => {
         // Use timeout to check status *after* state has updated
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
     // Cleanup polling when session ID changes or component unmounts
    return () => {
        stopPolling();
    };
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
    setCurrentSessionId(null); // Deselect current session
    setCurrentSessionDetails(null);
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
        setCurrentSessionId(response.session_id); // Select the new session automatically
    } catch (err: any) {
        console.error("Clustering start error:", err);
        const errorMsg = err.message || 'Не удалось запустить кластеризацию.';
        setError(errorMsg);
        toast.error(`Ошибка запуска кластеризации: ${errorMsg}`);
        throw err; // Rethrow so control knows it failed
    } finally {
        setIsLoading(false);
    }
  }, [fetchWithAuth, stopPolling]);

  // --- Delete Handler ---
   const handleDeleteAndRedistributeCluster = useCallback((clusterId: string | number) => {
    if (!currentSessionId || !currentSessionDetails) return;
    const clusterToDelete = currentSessionDetails.clusters.find(c => String(c.id) === String(clusterId));
    const clusterDisplayName = clusterToDelete?.name || `Кластер ${clusterId}`;
    setConfirmDeleteArgs({ clusterId, clusterDisplayName });
    setIsConfirmDeleteModalOpen(true);
  }, [currentSessionId, currentSessionDetails]);

  // --- Delete Confirmation ---
  const confirmDeletion = useCallback(async () => {
      if (!currentSessionId || !confirmDeleteArgs) return;
      const { clusterId, clusterDisplayName } = confirmDeleteArgs;
      setIsConfirmDeleteLoading(true);
      setIsDeletingId(clusterId);
      setError(null);
      stopPolling();
      toast.info(`Удаляем '${clusterDisplayName}' и перераспределяем...`);
      try {
          const response = await deleteAndRedistributeCluster(fetchWithAuth, currentSessionId, String(clusterId));
          toast.success(response.message || `'${clusterDisplayName}' удален.`);
          await fetchSessionResults(currentSessionId); // Refresh results
          // No need to update session list here, fetchSessionResults does it
      } catch (err: any) {
          console.error(`Error deleting/redistributing cluster ${clusterId}:`, err);
          const errorMsg = err.message || `Не удалось удалить/перераспределить '${clusterDisplayName}'.`;
          setError(errorMsg); // Show error in dashboard
          toast.error(errorMsg);
          fetchSessionResults(currentSessionId); // Try refresh even on error
      } finally {
          setIsConfirmDeleteLoading(false);
          setIsDeletingId(null);
          setIsConfirmDeleteModalOpen(false);
          setConfirmDeleteArgs(null);
          setSelectedClusterIds(new Set()); // Clear selection after delete
      }
  }, [currentSessionId, confirmDeleteArgs, fetchWithAuth, stopPolling, fetchSessionResults]);

  const closeConfirmDeleteModal = () => {
       setIsConfirmDeleteModalOpen(false);
       setConfirmDeleteArgs(null);
   };

  // --- Rename Handler ---
  const handleRenameCluster = useCallback(async (clusterId: string | number, newName: string): Promise<boolean> => {
    if (!currentSessionId) { toast.error("Нет активной сессии."); return false; }
    const clusterToRename = currentSessionDetails?.clusters.find(c => String(c.id) === String(clusterId));
    const oldDisplayName = clusterToRename?.name || `Кластер ${clusterId}`;
    setIsRenamingId(clusterId);
    setError(null);
    let success = false;
    try {
        await renameCluster(fetchWithAuth, currentSessionId, clusterId, newName);
        toast.success(`Кластер '${oldDisplayName}' переименован.`);
        await fetchSessionResults(currentSessionId); // Refresh results
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

  // --- Merge Handlers ---
  const handleToggleClusterSelection = useCallback((clusterId: string | number) => {
      setSelectedClusterIds(prev => {
          const newSet = new Set(prev);
          if (newSet.has(clusterId)) {
              newSet.delete(clusterId);
          } else {
              newSet.add(clusterId);
          }
          return newSet;
      });
  }, []);

  const handleMergeClick = useCallback(async () => {
      if (!currentSessionId || selectedClusterIds.size < 2) return;
      setIsMerging(true);
      setError(null);
      stopPolling();
      const idsToMerge = Array.from(selectedClusterIds);
      toast.info(`Слияние кластеров: ${idsToMerge.join(', ')}...`);
      try {
          const response = await mergeSelectedClusters(fetchWithAuth, currentSessionId, idsToMerge);
          toast.success(response.message || `Кластеры слиты.`);
          await fetchSessionResults(currentSessionId); // Refresh results
          setSelectedClusterIds(new Set()); // Clear selection
      } catch (err: any)
       {
           console.error(`Error merging clusters ${idsToMerge}:`, err);
           const errorMsg = err.message || `Не удалось слить кластеры.`;
           setError(errorMsg);
           toast.error(errorMsg);
           fetchSessionResults(currentSessionId); // Try refresh
       } finally {
          setIsMerging(false);
      }
  }, [currentSessionId, selectedClusterIds, fetchWithAuth, stopPolling, fetchSessionResults]);


  // --- Split Handlers ---
   const handleInitiateSplit = useCallback((clusterId: string | number) => {
        if (!currentSessionDetails) return;
        const clusterToSplit = currentSessionDetails.clusters.find(c => String(c.id) === String(clusterId));
        if (clusterToSplit) {
             setSplitTarget({ id: clusterId, name: clusterToSplit.name || `Кластер ${clusterId}` });
        }
   }, [currentSessionDetails]);

   const handleCancelSplit = useCallback(() => {
       setSplitTarget(null);
   }, []);

   const handleConfirmSplit = useCallback(async (numSplits: number) => {
        if (!currentSessionId || !splitTarget) return;
        setIsSplitting(true);
        setError(null);
        stopPolling();
        toast.info(`Разделение кластера '${splitTarget.name}' на ${numSplits} части...`);
        try {
            const response = await splitSelectedCluster(fetchWithAuth, currentSessionId, splitTarget.id, numSplits);
            toast.success(response.message || `Кластер разделен.`);
            setSplitTarget(null); // Close modal
            await fetchSessionResults(currentSessionId); // Refresh results
        } catch (err: any) {
             console.error(`Error splitting cluster ${splitTarget.id}:`, err);
             const errorMsg = err.message || `Не удалось разделить кластер.`;
             setError(errorMsg);
             toast.error(errorMsg);
             // Keep modal open on error? Or close? Let's close it.
             setSplitTarget(null);
             fetchSessionResults(currentSessionId); // Try refresh
         } finally {
            setIsSplitting(false);
        }
   }, [currentSessionId, splitTarget, fetchWithAuth, stopPolling, fetchSessionResults]);

  // Derived state for disabling controls
  const canAdjustClusters = currentSessionDetails?.status === 'SUCCESS' || currentSessionDetails?.status === 'RECLUSTERED';
  const overallDisabled = isProcessingAny || !canAdjustClusters;

  return (
    <div className="clustering-dashboard">
      <h2>Панель управления кластеризацией</h2>

      <SessionSelector
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        disabled={isProcessingAny} // Disable selection during any operation
        isLoading={isFetchingSessions}
        error={error && !currentSessionId ? error : null} // Show general error if no session selected
      />

      <ClusteringControls
        onStartClustering={handleStartClustering}
        disabled={isProcessingAny} // Disable start if anything is running
      />

      {currentSessionId && isFetchingResults && !pollingIntervalRef.current && ( // Show loading only on initial fetch
        <div className="card status-card"> <p>Загрузка результатов для сессии {currentSessionId.substring(0,8)}...</p> </div>
      )}
      {currentSessionId && !isFetchingResults && !currentSessionDetails && error && ( // Show error if fetch failed
           <div className="card status-card error-message"><p>Ошибка загрузки результатов: {error}</p></div>
       )}

      {/* Display area only if details are loaded */}
      {currentSessionId && currentSessionDetails && (
          <>
              <SessionDetailsDisplay details={currentSessionDetails} />

              <ExportControls
                    sessionId={currentSessionId}
                    fetchWithAuth={fetchWithAuth}
                    disabled={isProcessingAny} // Disable export during operations
                    sessionStatus={currentSessionDetails.status}
              />

              {(currentSessionDetails.status === 'SUCCESS' || currentSessionDetails.status === 'RECLUSTERED' || currentSessionDetails.status === 'PROCESSING') && (
                  <ChartsDisplay
                      details={currentSessionDetails}
                      sessionId={currentSessionId}
                  />
              )}

              {/* Merge Button - Render above the grid */}
              {canAdjustClusters && currentSessionDetails.clusters.length > 0 && (
                  <div className="card" style={{ padding: '1rem', marginBottom: '1rem', textAlign: 'center' }}>
                      <button
                          className="primary-btn"
                          onClick={handleMergeClick}
                          disabled={selectedClusterIds.size < 2 || overallDisabled}
                          title={selectedClusterIds.size < 2 ? "Выберите 2 или более кластера для слияния" : (overallDisabled ? "Операция недоступна" : "Слить выбранные кластеры")}
                      >
                          Слить выбранные ({selectedClusterIds.size})
                      </button>
                  </div>
              )}

              {/* Contact Sheets Grid */}
               {(currentSessionDetails.status === 'SUCCESS' || currentSessionDetails.status === 'RECLUSTERED') && (
                  <ContactSheetsGrid
                    clusters={currentSessionDetails.clusters || []}
                    sessionId={currentSessionId}
                    onRedistribute={handleDeleteAndRedistributeCluster}
                    onRename={handleRenameCluster}
                    isDeletingId={isDeletingId}
                    disabled={overallDisabled} // Pass combined disabled state
                    status={currentSessionDetails.status}
                    selectedIds={selectedClusterIds} // Pass selection state
                    onToggleSelection={handleToggleClusterSelection} // Pass handlers
                    onInitiateSplit={handleInitiateSplit} // Pass handlers
                  />
              )}

              {/* Handling other statuses */}
               {currentSessionDetails.status !== 'SUCCESS' && currentSessionDetails.status !== 'RECLUSTERED' && currentSessionDetails.status !== 'PROCESSING' && (
                 <div className="card status-card">
                     <p>Статус сессии {currentSessionId.substring(0,8)}...: <strong>{currentSessionDetails.status}</strong>.</p>
                     {currentSessionDetails.message && <p>{currentSessionDetails.message}</p>}
                     {currentSessionDetails.error && <p className="error-message">{currentSessionDetails.error}</p>}
                     <p>Результаты не могут быть отображены или изменены в данный момент.</p>
                 </div>
              )}
              {/* Handling processing status */}
              {currentSessionDetails.status === 'PROCESSING' && !isFetchingResults && (
                   <div className="card status-card">
                       <p>Сессия {currentSessionId.substring(0,8)}... обрабатывается. Статус будет обновлен автоматически.</p>
                       {currentSessionDetails.message && <p>{currentSessionDetails.message}</p>}
                   </div>
              )}
          </>
       )}

       {/* Placeholder messages when no session is selected/active */}
       {!currentSessionId && !isProcessingAny && sessions.length > 0 && ( <div className="card status-card"> <p>Выберите сессию из списка выше для просмотра результатов или запустите новую кластеризацию.</p> </div> )}
       {!currentSessionId && !isProcessingAny && sessions.length === 0 && !isFetchingSessions && !error && ( <div className="card status-card"> <p>Нет доступных сессий. Запустите новую кластеризацию, используя форму выше.</p> </div> )}

        {/* Delete Confirmation Modal */}
        <ConfirmationModal
            isOpen={isConfirmDeleteModalOpen}
            onClose={closeConfirmDeleteModal}
            onConfirm={confirmDeletion}
            title="Подтверждение удаления"
            message={
                confirmDeleteArgs ? (
                    <>Вы уверены, что хотите удалить <br /><strong>'{confirmDeleteArgs.clusterDisplayName}'</strong>?<br />Его точки будут перераспределены. Это действие необратимо.</>
                ) : ('Подтвердите действие.')
            }
            confirmText="Удалить и перераспределить"
            cancelText="Отмена"
            confirmButtonClass="danger-btn"
            isLoading={isConfirmDeleteLoading}
        />

        {/* Split Cluster Modal */}
        <SplitClusterModal
            isOpen={splitTarget !== null}
            onClose={handleCancelSplit}
            onConfirm={handleConfirmSplit}
            clusterId={splitTarget?.id ?? null}
            clusterDisplayName={splitTarget?.name ?? ''}
            isLoading={isSplitting}
        />

    </div>
  );
};

export default ClusteringDashboard;