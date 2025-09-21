// frontend/src/services/api.js
import axios from 'axios';

// В Docker-сети имя сервиса 'backend'
const API_BASE_URL = process.env.NODE_ENV === 'development'
    ? 'http://localhost:8000'
    : 'http://backend:8000';

export const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
});

// Обработчик ошибок
api.interceptors.response.use(
    response => response,
    error => {
        const message = error.response?.data?.detail || error.message || 'Unknown error';
        throw new Error(`Server error: ${message}`);
    }
);

export const predictFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
};

export const batchPredict = async (files) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    const response = await api.post('/api/batch_predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data; // Возвращаем JSON с результатами
};

export const downloadReport = async () => {
    const response = await api.get('/api/download_report', { responseType: 'blob' }); // ✅ Исправлено: /api/
    return response;
};
