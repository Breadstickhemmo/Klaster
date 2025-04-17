import React, { useState, useEffect } from 'react';
import '../styles/ContactSheet.css';
import { toast } from 'react-toastify';

interface ContactSheetProps {
  clusterId: string | number;
  imageUrl: string | null | undefined;
  clusterSize: number;
  initialName: string | null;
  onRedistribute: (clusterId: string | number) => void;
  onRename: (clusterId: string | number, newName: string) => Promise<boolean>;
  isProcessing: boolean;
}

const ImagePlaceholder: React.FC = () => (
    <div className="contact-sheet-image placeholder-image-container">
        <p>Нет<br/>отпечатка</p>
    </div>
);

const ContactSheet: React.FC<ContactSheetProps> = ({
  clusterId,
  imageUrl,
  clusterSize,
  initialName,
  onRedistribute,
  onRename,
  isProcessing
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [currentName, setCurrentName] = useState(initialName || '');
  const [isSavingName, setIsSavingName] = useState(false);

  useEffect(() => {
    setCurrentName(initialName || '');
    if (!isSavingName) {
        setIsEditing(false);
    }
  }, [initialName, isSavingName]);

  const handleRenameClick = () => {
    setIsEditing(true);
  };

  const handleCancelClick = () => {
    setIsEditing(false);
    setCurrentName(initialName || '');
  };

  const handleSaveClick = async () => {
    if (currentName === initialName) {
        setIsEditing(false);
        return;
    }
    setIsSavingName(true);
    try {
        const success = await onRename(clusterId, currentName.trim());
        if (success) {
            toast.success(`Кластер ${clusterId} переименован.`);
            setIsEditing(false);
        } else {
        }
    } catch (error) {
        console.error("Error saving cluster name:", error);
        toast.error("Не удалось сохранить имя кластера.");
    } finally {
        setIsSavingName(false);
    }
  };

  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentName(event.target.value);
  };

  const displayClusterName = initialName || `Кластер ${clusterId}`;
  const areActionsDisabled = isProcessing || isEditing || isSavingName;

  return (
    <div className={`contact-sheet-card ${isProcessing ? 'is-deleting' : ''} ${isEditing ? 'is-editing-name' : ''}`}>

      {isEditing ? (
        <div className="cluster-name-edit">
            <input
                type="text"
                value={currentName}
                onChange={handleNameChange}
                placeholder={`Имя для кластера ${clusterId}`}
                disabled={isSavingName}
                className="cluster-name-input"
                maxLength={100}
                autoFocus
            />
        </div>
      ) : (
        <h4 title={displayClusterName}>{displayClusterName}</h4>
      )}

      {imageUrl ? (
        <img
          src={imageUrl}
          alt={`Контактный отпечаток для кластера ${clusterId}`}
          className="contact-sheet-image"
          loading="lazy"
          onError={(e) => { /* ... обработка ошибки ... */ }}
         />
      ) : (
         <ImagePlaceholder />
      )}

      <p>Размер: {clusterSize} изображений</p>

      <div className="cluster-actions">
        {isEditing ? (
            <>
                <button
                    className="primary-btn save-name-btn"
                    onClick={handleSaveClick}
                    disabled={isSavingName}
                >
                    {isSavingName ? 'Сохр...' : 'Сохранить'}
                </button>
                <button
                    className="secondary-btn cancel-name-btn"
                    onClick={handleCancelClick}
                    disabled={isSavingName}
                >
                    Отмена
                </button>
            </>
        ) : (
             <>
                <button
                    className="secondary-btn rename-cluster-btn"
                    onClick={handleRenameClick}
                    disabled={areActionsDisabled}
                    title="Переименовать кластер"
                >
                    Переименовать
                </button>
                <button
                    className="secondary-btn delete-sheet-btn"
                    onClick={() => onRedistribute(clusterId)}
                    disabled={areActionsDisabled}
                    title={isProcessing ? "Выполняется операция..." : `Удалить кластер ${clusterId} и перераспределить его точки`}
                >
                    {isProcessing && !isEditing && !isSavingName ? 'Обработка...' : 'Удалить и перераспределить'}
                </button>
             </>
        )}
      </div>
    </div>
  );
};

export default ContactSheet;