// frontend/src/components/ResultsTable.jsx
import React from 'react';
import ReportDownloader from './ReportDownloader';

const ResultsTable = ({ results }) => {
  if (!results || results.length === 0) {
    return <p>Результаты не загружены. Загрузите ZIP-файлы.</p>;
  }

  return (
    <div style={{ marginTop: '30px' }}>
      <h2>📊 Результаты анализа ({results.length} исследований)</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px' }}>
        <thead>
          <tr style={{ backgroundColor: '#f0f0f0' }}>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Файл</th>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Study UID</th>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Series UID</th>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Вероятность патологии</th>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Класс</th>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Время (с)</th>
            <th style={{ border: '1px solid #ccc', padding: '8px' }}>Статус</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, idx) => (
            <tr key={idx} style={{ backgroundColor: result.pathology === 1 ? '#ffebee' : '#e8f5e8' }}>
              <td style={{ border: '1px solid #ccc', padding: '8px', wordBreak: 'break-all' }}>
                {result.path_to_study.split('/').pop()}
              </td>
              <td style={{ border: '1px solid #ccc', padding: '8px', wordBreak: 'break-all' }}>
                {result.study_uid}
              </td>
              <td style={{ border: '1px solid #ccc', padding: '8px', wordBreak: 'break-all' }}>
                {result.series_uid}
              </td>
              <td style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'right' }}>
                {result.probability_of_pathology.toFixed(6)}
              </td>
              <td style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>
                {result.pathology === 1 ? 'Патология' : 'Норма'}
              </td>
              <td style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'right' }}>
                {result.time_of_processing.toFixed(2)}
              </td>
              <td style={{ border: '1px solid #ccc', padding: '8px', textAlign: 'center' }}>
                {result.processing_status}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <ReportDownloader />
    </div>
  );
};

export default ResultsTable;
