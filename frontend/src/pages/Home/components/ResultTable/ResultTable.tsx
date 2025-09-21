import { type FC } from 'react';
import {
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableData
} from '@ui/Table';

const results = [{
  filename: 'test.zip',
  file_name: 'test2.zip',
  study_uid: 'study_uid',
  series_uid: 'series_uid',
  probability_of_pathology: 0.2,
  pathology: 1,
  time_of_processing: 12,
  processing_time_sec: 5,
  processing_status: 'Success'
}];

export const ResultTable: FC = () => {
  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeader>Файл</TableHeader>
          <TableHeader>Study UID</TableHeader>
          <TableHeader>Series UID</TableHeader>
          <TableHeader>Вероятность патологии</TableHeader>
          <TableHeader>Класс</TableHeader>
          <TableHeader>Время (с)</TableHeader>
          <TableHeader>Статус</TableHeader>
        </TableRow>
      </TableHead>
      <TableBody>
        {results.map((result, idx) => (
          <TableRow key={idx}>
            <TableData>
              {result.file_name || result.filename || 'N/A'}
            </TableData>
            <TableData>
              {result.study_uid || 'N/A'}
            </TableData>
            <TableData>
              {result.series_uid || 'N/A'}
            </TableData>
            <TableData>
              {typeof result.probability_of_pathology === 'number' ? result.probability_of_pathology.toFixed(4) : 'N/A'}
            </TableData>
            <TableData>
              {result.pathology === 1 ? 'Патология' : result.pathology === 0 ? 'Норма' : 'N/A'}
            </TableData>
            <TableData>
              {(result.time_of_processing || result.processing_time_sec || 0).toFixed(2)}
            </TableData>
            <TableData>
              {result.processing_status || 'N/A'}
            </TableData>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};
