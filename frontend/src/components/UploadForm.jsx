// frontend/src/components/UploadForm.jsx
import React, { useState } from 'react';
import { batchPredict, downloadReport } from '../services/api'; // ✅ Импортируем downloadReport
import ResultsTable from './ResultsTable';

const UploadForm = () => {
    const [files, setFiles] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [results, setResults] = useState([]);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setFiles(Array.from(e.target.files));
        setError(null);
    };

    const handleSubmit = async () => {
        if (!files.length) return;
        setIsProcessing(true);
        setError(null);

        try {
            // ✅ Только получаем результаты — НЕ скачиваем файл
            const response = await batchPredict(files);
            
            if (!response || !response.results) {
                throw new Error("Invalid response from server");
            }

            setResults(response.results); // ✅ Заполняем таблицу
            alert('✅ Обработка завершена! Результаты отображены ниже.');

        } catch (err) {
            setError(err.message);
            alert(`❌ Ошибка: ${err.message}`);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
            <h1>📌 NORMSCAN — ИИ-сервис для выявления КТ ОГК с «нормой»</h1>
            {error && <div style={{color: 'red', marginBottom: '10px'}}>{error}</div>}
            <p>Загрузите ZIP-архивы с DICOM-файлами для анализа.</p>
            <input type="file" multiple accept=".zip" onChange={handleFileChange} />
            <button 
                onClick={handleSubmit} 
                disabled={isProcessing || !files.length}
                style={{ 
                    marginLeft: '10px', 
                    padding: '10px 20px', 
                    backgroundColor: '#007BFF', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '4px', 
                    cursor: 'pointer' 
                }}
            >
                {isProcessing ? 'Обработка...' : 'Загрузить и проанализировать'}
            </button>
            <ResultsTable results={results} /> {/* ✅ Таблица + кнопка скачивания внутри */}
        </div>
    );
};

export default UploadForm;
