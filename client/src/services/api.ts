type FetchWithAuth = (url: string, options?: RequestInit) => Promise<Response>;

export interface StartClusteringPayload {
    algorithm: 'kmeans' | 'dbscan';
    params: { [key: string]: number };
    embeddingFile: File;
}

export interface ClusterResult {
    id: string | number;
    original_id: string | null;
    name: string | null;
    size: number;
    contactSheetUrl: string | null;
    metrics?: any;
    centroid_2d?: [number, number] | null;
}

export interface SessionResultResponse {
    session_id: string;
    status: string;
    original_filename: string | null;
    algorithm: string;
    params: { [key: string]: number | string };
    num_clusters: number | null;
    processing_time_sec: number | null;
    clusters: ClusterResult[];
    scatter_data?: ScatterPoint[] | { error: string } | null;
    scatter_pca_time_sec?: number | null;
    message?: string;
    error?: string;
}

export interface SessionListItem {
     session_id: string;
     created_at: string;
     status: string;
     algorithm: string;
     params: { [key: string]: number | string };
     num_clusters: number | null;
     result_message: string | null;
     original_filename: string | null;
}

export interface ScatterPoint {
    x: number;
    y: number;
    cluster: string;
}

const handleResponse = async (response: Response) => {
    if (response.status === 204) {
        return null;
    }
    let data;
    try {
        data = await response.json();
    } catch (e) {
        if (!response.ok) {
             throw new Error(`HTTP error! Status: ${response.status}, Body is not JSON.`);
        }
        console.warn("Response was OK but body is not JSON.");
        return null;
    }
    if (!response.ok) {
        throw new Error(data?.error || data?.message || `HTTP error! Status: ${response.status}`);
    }
    return data;
};

export const startClustering = async (fetchWithAuth: FetchWithAuth, data: StartClusteringPayload): Promise<{ session_id: string }> => {
    const formData = new FormData();
    formData.append('embeddingFile', data.embeddingFile);
    formData.append('algorithm', data.algorithm);
    formData.append('params', JSON.stringify(data.params));
    const response = await fetchWithAuth('/api/clustering/start', { method: 'POST', body: formData });
    return handleResponse(response);
};

export const getClusteringSessions = async (fetchWithAuth: FetchWithAuth): Promise<SessionListItem[]> => {
    const response = await fetchWithAuth('/api/clustering/sessions');
    return handleResponse(response);
};

export const getClusteringResults = async (fetchWithAuth: FetchWithAuth, sessionId: string): Promise<SessionResultResponse> => {
    const response = await fetchWithAuth(`/api/clustering/results/${sessionId}`);
    const results: SessionResultResponse = await handleResponse(response);
    return results;
};

export const deleteClusterAndRecluster = async (fetchWithAuth: FetchWithAuth, sessionId: string, clusterLabel: string | number): Promise<{ message: string; new_session_id: string }> => {
    const response = await fetchWithAuth(`/api/clustering/results/${sessionId}/cluster/${clusterLabel}`, { method: 'DELETE' });
    return handleResponse(response);
};

export const renameCluster = async (fetchWithAuth: FetchWithAuth, sessionId: string, clusterId: string | number, newName: string): Promise<{ message: string; cluster: { id: string | number, name: string | null }}> => {
    const response = await fetchWithAuth(`/api/clustering/results/${sessionId}/adjust`, {
        method: 'POST',
        body: JSON.stringify({ action: 'RENAME', cluster_id: clusterId, new_name: newName })
    });
    return handleResponse(response);
};