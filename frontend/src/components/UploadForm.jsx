// frontend/src/components/UploadForm.jsx
import React, { useState } from 'react';
import api from '../services/api';
import ResultsTable from './ResultsTable';

const UploadForm = () => {
  const [files, setFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState([]);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleSubmit = async () => {
    if (!files.length) return;

    setIsProcessing(true);
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
      const response = await axios.post('/batch_predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data.results);
      alert('Обработка завершена! Отчёт сгенерирован.');
    } catch (error) {
      alert('Ошибка: ' + error.response?.data?.detail || error.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>📌 NORMSCAN — ИИ-сервис для выявления КТ ОГК с «нормой»</h1>
      <p>Загрузите ZIP-архивы с DICOM-файлами для анализа.</p>

      <input type="file" multiple accept=".zip" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={isProcessing || !files.length} style={{ marginLeft: '10px', padding: '10px 20px' }}>
        {isProcessing ? 'Обработка...' : 'Загрузить и проанализировать'}
      </button>

      <ResultsTable results={results} />
    </div>
  );
};

export default UploadForm;
