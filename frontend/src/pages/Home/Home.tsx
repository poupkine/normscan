import { type FC, useEffect } from 'react';
import { useAppDispatch } from '@store/hooks';
import { setErrorMessage } from '@store/slices/errorSlice';
import { useApi } from '@hooks/useApi';
import { getReportList } from '@api/reports';
import { Loader } from '@ui/Loader';
import { InfoCard } from './components/InfoCard';
import { ActionCard } from './components/ActionCard';
import { ResultTable } from './components/ResultTable';
import styles from './Home.module.css';

export const Home: FC = () => {
  const dispatch = useAppDispatch();
  const {
    status,
    data,
    error,
    isFetching,
    sendRequest
  } = useApi(getReportList);

  useEffect(() => {
    sendRequest();
  }, [])

  useEffect(() => {
    if (status === 'success' && data) {
      console.log(data);
    } else if (status === 'error' && error) {
      dispatch(setErrorMessage(error.getErrorMessage()));
    }

  }, [status, data, error]);

  return (
    <div className={styles['home-page']}>
      {isFetching && <Loader />}
      <h1 className={styles['home-page__title']}>
        NORMSCAN — ИИ-сервис для выявления КТ ОГК с «нормой»
      </h1>
      <div className={styles['home-page__content-grid']}>
        <InfoCard />
        <ActionCard />
      </div>
      <h2 className={styles['home-page__subtitle']}>Результаты</h2>
      <ResultTable />
    </div>
  );
};
