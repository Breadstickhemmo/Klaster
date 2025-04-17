import React, { useMemo } from 'react';
import { SessionResultResponse, ClusterResult } from '../services/api';
import { Bar, Scatter } from 'react-chartjs-2';
import { ChartOptions, ChartData } from 'chart.js';
import '../styles/ChartsDisplay.css';

interface ChartsDisplayProps {
    details: SessionResultResponse | null;
    sessionId: string | null;
}

const generateColor = (index: number, total: number, saturation = 70, lightness = 60): string => {
  const hue = (index * (360 / Math.max(total, 1))) % 360;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
};

const OUTLIER_COLOR = 'rgba(150, 150, 150, 0.5)';

const sortClustersNumerically = <T extends { id: string | number }>(clusters: T[]): T[] => {
    return [...clusters].sort((a, b) => {
        const numA = parseFloat(String(a.id));
        const numB = parseFloat(String(b.id));
        if (isNaN(numA) && isNaN(numB)) return String(a.id).localeCompare(String(b.id));
        if (isNaN(numA)) return 1;
        if (isNaN(numB)) return -1;
        return numA - numB;
    });
};

const ChartsDisplay: React.FC<ChartsDisplayProps> = ({ details, sessionId }) => {

    const { barChartData, centroidChartData, scatterChartData, scatterError } = useMemo(() => {
        const result: {
            barChartData: ChartData<'bar'> | null;
            centroidChartData: ChartData<'scatter'> | null;
            scatterChartData: ChartData<'scatter'> | null;
            scatterError: string | null;
        } = { barChartData: null, centroidChartData: null, scatterChartData: null, scatterError: null };

        if (!details || (details.status !== 'SUCCESS' && details.status !== 'RECLUSTERED' && details.status !== 'PROCESSING')) {
             if (details?.status !== 'PROCESSING') { return result; }
        }

        if (details.clusters && details.clusters.length > 0) {
            const sortedClustersForBar = sortClustersNumerically<ClusterResult>(details.clusters);
            const barLabels = sortedClustersForBar.map(c => `Кластер ${c.id}`);
            const barDataPoints = sortedClustersForBar.map(c => c.size);
            result.barChartData = { labels: barLabels, datasets: [{ label: 'Размер кластера (кол-во изображений)', data: barDataPoints, backgroundColor: 'rgba(0, 123, 255, 0.6)', borderColor: 'rgba(0, 123, 255, 1)', borderWidth: 1, }], };
        }

        const clustersWith2dCentroids = details.clusters?.filter(c => c.centroid_2d && c.centroid_2d.length === 2) || [];
        if (clustersWith2dCentroids.length > 0) {
            const sortedCentroids = sortClustersNumerically<ClusterResult>(clustersWith2dCentroids);
            result.centroidChartData = {
                datasets: sortedCentroids.map((cluster, index) => ({
                    label: `Центр ${cluster.id}`,
                    data: [{ x: cluster.centroid_2d![0], y: cluster.centroid_2d![1] }],
                    backgroundColor: generateColor(index, sortedCentroids.length),
                    pointRadius: 6,
                    pointHoverRadius: 8,
                })),
            };
        }

        if (details.scatter_data) {
            if (Array.isArray(details.scatter_data)) {
                const points = details.scatter_data;
                const clustersPresent = Array.from(new Set(points.map(p => p.cluster))).sort((a, b) => {
                     const numA = parseInt(a); const numB = parseInt(b);
                     if (a === '-1') return -1;
                     if (b === '-1') return 1;
                     if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
                     return a.localeCompare(b);
                });

                const maxClusterNum = clustersPresent.reduce((max, id) => {
                    if (id === '-1') return max;
                    const numId = parseInt(id);
                    return !isNaN(numId) ? Math.max(max, numId) : max;
                 }, -1);
                const numColorsNeeded = maxClusterNum + 1;

                result.scatterChartData = {
                    datasets: clustersPresent.map((clusterId) => {
                        const clusterPoints = points
                            .filter(p => p.cluster === clusterId)
                            .map(p => ({ x: p.x, y: p.y }));

                        const isOutlier = clusterId === '-1';
                        const clusterIndex = isOutlier ? -1 : parseInt(clusterId);

                        return {
                            label: isOutlier ? 'Выбросы (-1)' : `Кластер ${clusterId}`,
                            data: clusterPoints,
                            backgroundColor: isOutlier ? OUTLIER_COLOR : generateColor(clusterIndex, numColorsNeeded, 60, 70),
                            pointRadius: 2.5,
                            pointHoverRadius: 4,
                            showLine: false,
                            borderColor: 'transparent'
                        };
                    }),
                };
            } else if (details.scatter_data.error) {
                result.scatterError = details.scatter_data.error;
            }
        } else if (details.status === 'SUCCESS' || details.status === 'RECLUSTERED') {
             result.scatterError = "Данные для графика рассеяния не были сгенерированы.";
        }

        return result;
    }, [details]);

    const barChartOptions = useMemo((): ChartOptions<'bar'> => ({ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' as const, }, title: { display: true, text: `Распределение размеров кластеров (${details?.algorithm?.toUpperCase() || 'N/A'})`, font: { size: 16 } }, tooltip: { callbacks: { label: function(context) { let label = context.dataset.label || ''; if (label) { label += ': '; } if (context.parsed.y !== null) { label += `${context.parsed.y} изображений`; } return label; } } } }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Количество изображений' } }, x: { title: { display: true, text: 'Номер кластера' } } }, }), [details?.algorithm]);
    const centroidChartOptions = useMemo((): ChartOptions<'scatter'> => ({ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' as const, display: (details?.clusters?.length ?? 0) <= 20, }, title: { display: true, text: 'Центроиды кластеров (PCA)', font: { size: 16 } }, tooltip: { callbacks: { label: function(context) { const label = context.dataset.label || ''; const point = context.parsed; return `${label}: (x: ${point.x.toFixed(2)}, y: ${point.y.toFixed(2)})`; } } } }, scales: { x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Компонента PCA 1' } }, y: { type: 'linear', title: { display: true, text: 'Компонента PCA 2' } } }, }), [details?.clusters]);

    const scatterChartOptions = useMemo((): ChartOptions<'scatter'> => ({
        responsive: true,
        maintainAspectRatio: false,
         plugins: {
            legend: {
                position: 'right' as const,
                display: (scatterChartData?.datasets?.length ?? 0) <= 15,
            },
            title: {
                display: true,
                text: 'График рассеяния изображений (PCA)',
                font: { size: 16 }
            },
            tooltip: {
                 callbacks: {
                    label: function(context) {
                        return context.dataset.label || '';
                    }
                }
            },
            zoom: {
               zoom: {
                 wheel: { enabled: true },
                 pinch: { enabled: true },
                 mode: 'xy',
               },
                pan: {
                    enabled: true,
                    mode: 'xy',
                },
            }
        },
        scales: {
             x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Компонента PCA 1' } },
             y: { type: 'linear', title: { display: true, text: 'Компонента PCA 2' } }
        },
    }), [scatterChartData]);

    const scatterDescription = useMemo(() => {
        if (!details) {
            return "Описание графика недоступно.";
        }
        const dataArray = Array.isArray(details.scatter_data) ? details.scatter_data : null;
        const pointCount = dataArray ? dataArray.length : 'N/A';
        let desc = `Визуализация сэмпла изображений (${pointCount} точек) в 2D пространстве после PCA, окрашенных по кластерам.`;
        desc += " Используйте колесо мыши/жесты для зума/перемещения.";
        if (details.scatter_pca_time_sec !== null && details.scatter_pca_time_sec !== undefined) {
            desc += ` Время расчета PCA: ${details.scatter_pca_time_sec.toFixed(2)} сек.`;
        }
        return desc;
    }, [details]);


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
                    {barChartData ? ( <div className="chart-container" style={{ height: '350px' }}> <Bar key={`${chartKeyBase}-bar`} options={barChartOptions} data={barChartData} /> </div> )
                    : (<p className='chart-placeholder-text'>Нет данных для графика.</p>)
                    }
                </div>

                <div className='chart-wrapper' style={{marginBottom: '2rem'}}>
                    <h4>График центроидов (PCA)</h4>
                    {centroidChartData ? ( <div className="chart-container" style={{ height: '400px' }}> <Scatter key={`${chartKeyBase}-centroid-scatter`} options={centroidChartOptions} data={centroidChartData} /> </div> )
                    : ( details.status === 'PROCESSING' ? <p className='chart-placeholder-text'>Расчет 2D координат...</p> : <p className='chart-placeholder-text'>Нет данных 2D центроидов.</p> )
                    }
                    <p className="chart-description">Визуализация центров кластеров в 2D пространстве после PCA.</p>
                </div>

                 <div className='chart-wrapper' style={{marginBottom: '2rem'}}>
                    <h4>График рассеяния изображений (PCA)</h4>
                    {scatterChartData ? (
                        <div className="chart-container" style={{ height: '500px' }}>
                            <Scatter key={`${chartKeyBase}-embedding-scatter`} options={scatterChartOptions} data={scatterChartData} />
                        </div>
                    ) : scatterError ? (
                         <p className='chart-placeholder-text error'>{scatterError}</p>
                    ) : (
                         <p className='chart-placeholder-text'>
                            {details.status === 'PROCESSING' ? 'Генерация данных...' : 'Данные для графика рассеяния недоступны.'}
                        </p>
                    )}
                    <p className="chart-description">{scatterDescription}</p>
                </div>
            </div>
        </div>
    );
};

export default ChartsDisplay;