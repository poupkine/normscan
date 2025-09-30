import { type FC } from 'react';
import type { ResultList } from '@pages/home/slice';
import {
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableData
} from '@ui/Table';

interface Props {
  resultList: ResultList;
}

export const ResultTable: FC<Props> = ({ resultList }) => {
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
        {resultList.map((result, idx) => (
          <TableRow key={idx}>
            <TableData>
              {result.filename || 'N/A'}
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
              {result.time_of_processing.toFixed(2)}
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
