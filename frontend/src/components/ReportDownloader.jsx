// frontend/src/components/ReportDownloader.jsx
import React from 'react';
import { api } from '../services/api'; // ✅ Исправлено: import { api }

const ReportDownloader = () => {
    const handleDownload = async () => {
        try {
            const response = await api.get('/download_report', { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'report.xlsx');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            alert('Не удалось скачать отчёт. Попробуйте обработать файлы снова.');
        }
    };

    return (
        <button
            onClick={handleDownload}
            style={{
                marginTop: '20px',
                padding: '12px 24px',
                fontSize: '16px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer'
            }}
        >
            📥 Скачать Excel-отчёт
        </button>
    );
};

export default ReportDownloader;
