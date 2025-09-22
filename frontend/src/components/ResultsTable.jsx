// frontend/src/components/ResultsTable.jsx
import React from 'react';
import { downloadReport } from '../services/api'; // ✅ Импортируем downloadReport

const ResultsTable = ({ results }) => {
    if (!results || results.length === 0) {
        return (
            <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px' }}>
                <p>📊 Результаты появятся здесь после обработки файлов.</p>
            </div>
        );
    }

    const handleDownload = async () => {
        try {
            const response = await downloadReport();
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'normscan_report.xlsx');
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            alert('✅ Отчёт успешно скачан!');
        } catch (error) {
            alert('❌ Не удалось скачать отчёт. Попробуйте позже.');
        }
    };

    return (
        <div style={{ marginTop: '30px' }}>
            <h2>📊 Результаты анализа ({results.length} исследований)</h2>
            <table style={{ 
                width: '100%', 
                borderCollapse: 'collapse', 
                marginTop: '10px', 
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)', 
                borderRadius: '8px', 
                overflow: 'hidden' 
            }}>
                <thead>
                    <tr style={{ backgroundColor: '#007BFF', color: 'white' }}>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Файл</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Study UID</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'left' }}>Series UID</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'right' }}>Вероятность патологии</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'center' }}>Класс</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'right' }}>Время (с)</th>
                        <th style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'center' }}>Статус</th>
                    </tr>
                </thead>
                <tbody>
                    {results.map((result, idx) => (
                        <tr key={idx} style={{ backgroundColor: idx % 2 === 0 ? '#ffffff' : '#f8f9fa' }}>
                            <td style={{ border: '1px solid #dee2e6', padding: '12px' }}>
                                {result.file_name || result.filename || 'N/A'}
                            </td>
                            <td style={{ border: '1px solid #dee2e6', padding: '12px' }}>
                                {result.study_uid || 'N/A'}
                            </td>
                            <td style={{ border: '1px solid #dee2e6', padding: '12px' }}>
                                {result.series_uid || 'N/A'}
                            </td>
                            <td style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'right' }}>
                                {typeof result.probability_of_pathology === 'number' ? result.probability_of_pathology.toFixed(4) : 'N/A'}
                            </td>
                            <td style={{ 
                                border: '1px solid #dee2e6', 
                                padding: '12px', 
                                textAlign: 'center', 
                                fontWeight: 'bold', 
                                color: result.pathology === 1 ? '#dc3545' : '#28a745' 
                            }}>
                                {result.pathology === 1 ? 'Патология' : result.pathology === 0 ? 'Норма' : 'N/A'}
                            </td>
                            <td style={{ border: '1px solid #dee2e6', padding: '12px', textAlign: 'right' }}>
                                {(result.time_of_processing || result.processing_time_sec || 0).toFixed(2)}
                            </td>
                            <td style={{ 
                                border: '1px solid #dee2e6', 
                                padding: '12px', 
                                textAlign: 'center', 
                                color: result.processing_status === 'Success' ? '#28a745' : '#dc3545' 
                            }}>
                                {result.processing_status || 'N/A'}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {/* ✅ Кнопка скачивания — только по клику */}
            <div style={{ marginTop: '30px', textAlign: 'center' }}>
                <button
                    onClick={handleDownload}
                    style={{
                        padding: '12px 32px',
                        fontSize: '18px',
                        backgroundColor: '#28a745',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        boxShadow: '0 3px 6px rgba(0,0,0,0.1)'
                    }}
                >
                    📥 Скачать Excel-отчёт
                </button>
            </div>
        </div>
    );
};

export default ResultsTable;
