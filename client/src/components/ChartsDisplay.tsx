import React, { useMemo } from 'react';
import { SessionResultResponse } from '../services/api';
import { Bar, Scatter } from 'react-chartjs-2';
import { ChartOptions, ChartData } from 'chart.js';
import '../styles/ChartsDisplay.css';

interface ChartsDisplayProps {
    details: SessionResultResponse | null;
    sessionId: string | null;
}

const generateColor = (index: number, total: number): string => {
  const hue = (index * (360 / Math.max(total, 1))) % 360;
  return `hsl(${hue}, 70%, 60%)`;
};
const PARAM_LABELS: { [key: string]: string } = {
    n_clusters: 'Количество кластеров',
    eps: 'Эпсилон (eps)',
    min_samples: 'Мин. изображений в кластере'
};


const ChartsDisplay: React.FC<ChartsDisplayProps> = ({ details, sessionId }) => {

    const { barChartData, centroidChartData } = useMemo(() => {
        const result: { barChartData: ChartData<'bar'> | null; centroidChartData: ChartData<'scatter'> | null; } = { barChartData: null, centroidChartData: null };
        if (!details?.clusters || details.clusters.length === 0 || (details.status !== 'SUCCESS' && details.status !== 'RECLUSTERED' && details.status !== 'PROCESSING')) {
             if (details?.status !== 'PROCESSING') { return result; }
        }
        const sortedClustersForBar = [...details.clusters].sort((a, b) => { const labelA = String(a.id); const labelB = String(b.id); const numA = parseFloat(labelA); const numB = parseFloat(labelB); if (!isNaN(numA) && !isNaN(numB)) { return numA - numB; } return labelA.localeCompare(labelB); });
        const barLabels = sortedClustersForBar.map(c => `Кластер ${c.id}`);
        const barDataPoints = sortedClustersForBar.map(c => c.size);
        result.barChartData = { labels: barLabels, datasets: [{ label: 'Размер кластера (кол-во изображений)', data: barDataPoints, backgroundColor: 'rgba(0, 123, 255, 0.6)', borderColor: 'rgba(0, 123, 255, 1)', borderWidth: 1, }], };
        const clustersWith2d = details.clusters.filter(c => c.centroid_2d && c.centroid_2d.length === 2);
        if (clustersWith2d.length > 0) { result.centroidChartData = { datasets: clustersWith2d.map((cluster, index) => ({ label: `Кластер ${cluster.id}`, data: [{ x: cluster.centroid_2d![0], y: cluster.centroid_2d![1] }], backgroundColor: generateColor(index, clustersWith2d.length), pointRadius: 5, pointHoverRadius: 7, })), }; }
        return result;
    }, [details]);
    const barChartOptions = useMemo((): ChartOptions<'bar'> => ({ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' as const, }, title: { display: true, text: `Распределение размеров кластеров (${details?.algorithm?.toUpperCase() || 'N/A'})`, font: { size: 16 } }, tooltip: { callbacks: { label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += `${context.parsed.y} изображений`; } return label; } } } }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Количество изображений' } }, x: { title: { display: true, text: 'Номер кластера' } } }, }), [details?.algorithm]);
    const centroidChartOptions = useMemo((): ChartOptions<'scatter'> => ({ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' as const, display: (details?.clusters?.length ?? 0) <= 20, }, title: { display: true, text: 'Центроиды кластеров (PCA)', font: { size: 16 } }, tooltip: { callbacks: { label: function(context) { const label = context.dataset.label || ''; const point = context.parsed; return `${label}: (x: ${point.x.toFixed(2)}, y: ${point.y.toFixed(2)})`; } } } }, scales: { x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Компонента PCA 1' } }, y: { type: 'linear', title: { display: true, text: 'Компонента PCA 2' } } }, }), [details?.clusters]);


    if (!details || (details.status !== 'SUCCESS' && details.status !== 'RECLUSTERED' && details.status !== 'PROCESSING')) {
        return null;
    }

    const chartKeyBase = sessionId || 'no-session';

    return (
        <div className="card charts-card">
            <h3 className="charts-main-title">Визуализации кластеризации</h3>
            <div className="charts-column-layout">

                <div className='chart-wrapper' style={{marginBottom: '2rem'}}>
                    <h4>Распределение размеров кластеров</h4>
                    {barChartData ? (
                        <div className="chart-container" style={{ height: '350px' }}>
                            <Bar key={`${chartKeyBase}-bar`} options={barChartOptions} data={barChartData} />
                        </div>
                     ) : (<p className='chart-placeholder-text'>Нет данных для графика.</p>)}
                     <p className="chart-description">Визуализация распределения изображений по кластерам.</p>
                </div>

                <div className='chart-wrapper' style={{marginBottom: '2rem'}}>
                    <h4>График центроидов (PCA)</h4>
                    {centroidChartData ? (
                        <div className="chart-container" style={{ height: '400px' }}>
                            <Scatter key={`${chartKeyBase}-scatter`} options={centroidChartOptions} data={centroidChartData} />
                        </div>
                     ) : ( details.status === 'PROCESSING' ? <p className='chart-placeholder-text'>Расчет 2D координат...</p> : <p className='chart-placeholder-text'>Нет данных 2D центроидов.</p> )}
                    <p className="chart-description">Визуализация центров кластеров в 2D пространстве после PCA.</p>
                </div>

                 <div className='chart-wrapper' style={{marginBottom: '2rem'}}> <h4>График рассеяния (PCA/t-SNE)</h4> <div className="graph-placeholder" style={{ height: '300px'}}> <img src={`https://placehold.co/300x150/E8E8E8/A9A9A9?text=Scatter+Plot`} alt="Scatter Plot Placeholder"/> <p>Визуализация кластеров в 2D. <br/>Требуется передача координат точек с бэкенда.</p> </div> </div>
                 <div className='chart-wrapper' style={{marginBottom: '2rem'}}> <h4>График силуэтов</h4> <div className="graph-placeholder" style={{ height: '300px'}}> <img src={`https://placehold.co/300x150/E8E8E8/A9A9A9?text=Silhouette+Plot`} alt="Silhouette Plot Placeholder"/> <p>Оценка качества кластеризации. <br/>Требуется расчет силуэта для каждой точки.</p> </div> </div>
                 {details.algorithm === 'kmeans' && (<div className='chart-wrapper' style={{marginBottom: '2rem'}}> <h4>График "Локтя" (Elbow Plot)</h4> <div className="graph-placeholder" style={{ height: '300px'}}> <img src={`https://placehold.co/300x150/E8E8E8/A9A9A9?text=Elbow+Plot`} alt="Elbow Plot Placeholder"/> <p>Помогает выбрать оптимальное K. <br/>Требуется запуск K-Means с разными K.</p> </div> </div> )}
                 {details.algorithm === 'dbscan' && (<div className='chart-wrapper' style={{marginBottom: '0'}}> <h4>Визуализация выбросов (DBSCAN)</h4> <div className="graph-placeholder" style={{ height: '300px'}}> <img src={`https://placehold.co/300x150/E8E8E8/A9A9A9?text=Outlier+Plot+(DBSCAN)`} alt="Outlier Plot Placeholder"/> <p>Отображение точек-выбросов (-1). <br/>Зависит от данных для Графика рассеяния.</p> </div> </div> )}

            </div>
        </div>
    );
};

export default ChartsDisplay;