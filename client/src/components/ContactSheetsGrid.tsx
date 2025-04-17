import React from 'react';
import ContactSheet from './ContactSheet';
import { ClusterResult } from '../services/api';
import '../styles/ContactSheetsGrid.css';
import '../styles/ContactSheet.css';

interface ContactSheetsGridProps {
    clusters: ClusterResult[];
    sessionId: string | null;
    onDelete: (clusterId: string | number) => void;
    isDeletingId: string | number | null;
    disabled: boolean;
    status: string;
}

const ContactSheetsGrid: React.FC<ContactSheetsGridProps> = ({
    clusters,
    sessionId,
    onDelete,
    isDeletingId,
    disabled,
    status
}) => {

    if ((status !== 'SUCCESS' && status !== 'RECLUSTERED')) {
        return null;
    }
    if (!clusters || clusters.length === 0) {
        if (status === 'SUCCESS' || status === 'RECLUSTERED') {
            return (
                <div className="card status-card">
                    <p>Кластеризация завершена, но не найдено кластеров для отображения контактных отпечатков.</p>
                </div>
            );
        }
        return null;
    }

    const sortedClusters = [...clusters].sort((a, b) => {
        const numA = parseFloat(String(a.id));
        const numB = parseFloat(String(b.id));

        if (isNaN(numA) && isNaN(numB)) {
             return String(a.id).localeCompare(String(b.id));
        }
        if (isNaN(numA)) return 1;
        if (isNaN(numB)) return -1;

        return numA - numB;
    });


    return (
        <div className="card contact-sheets-card">
            <h3>Контактные отпечатки ({sortedClusters.length} шт.)</h3>
            <div className="contact-sheets-grid-layout">
                {sortedClusters.map(cluster => cluster.contactSheetUrl ? (
                    <ContactSheet
                        key={cluster.id}
                        clusterId={cluster.id}
                        imageUrl={cluster.contactSheetUrl}
                        clusterSize={cluster.size}
                        onDelete={onDelete}
                        isDeleting={isDeletingId === cluster.id || disabled}
                    />
                ) : (
                    <div key={cluster.id} className="contact-sheet-card placeholder">
                        <h4>Кластер {cluster.id}</h4>
                        <div className="placeholder-image">
                            <p>Нет<br />отпечатка</p>
                        </div>
                        <p>Размер: {cluster.size} изображений</p>
                        <button className="secondary-btn delete-sheet-btn" disabled>Удаление недоступно</button>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ContactSheetsGrid;