import React from 'react';
import '../styles/ContactSheet.css';

interface ContactSheetProps {
  clusterId: string | number;
  imageUrl: string | null | undefined;
  clusterSize: number;
  onRedistribute: (clusterId: string | number) => void;
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
  onRedistribute,
  isProcessing
}) => {
  const buttonText = isProcessing ? 'Обработка...' : 'Удалить и перераспределить';
  const buttonTitle = isProcessing ? "Выполняется операция..." : `Удалить кластер ${clusterId} и перераспределить его точки`;

  const cardClassName = `contact-sheet-card ${isProcessing ? 'is-deleting' : ''}`;

  return (
    <div className={cardClassName}>
      <h4>Кластер {clusterId}</h4>

      {imageUrl ? (
        <img
          src={imageUrl}
          alt={`Контактный отпечаток для кластера ${clusterId}`}
          className="contact-sheet-image"
          loading="lazy"
          onError={(e) => {
              const target = e.target as HTMLImageElement;
              target.onerror = null;
              target.alt = `Ошибка загрузки: ${clusterId}`;
              target.style.display = 'none';
              console.warn(`Failed to load contact sheet image: ${imageUrl}`);
          }}
         />
      ) : (
         <ImagePlaceholder />
      )}

      <p>Размер: {clusterSize} изображений</p>
      <button
        className="secondary-btn delete-sheet-btn"
        onClick={() => onRedistribute(clusterId)}
        disabled={isProcessing}
        title={buttonTitle}
        aria-label={`Удалить кластер ${clusterId} и перераспределить точки`}
      >
        {buttonText}
      </button>
    </div>
  );
};

export default ContactSheet;