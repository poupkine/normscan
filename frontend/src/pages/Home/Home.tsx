import { type FC, useEffect } from 'react';
import { useApi } from '@hooks/useApi';
import { getReportList } from '@api/reports';
import styles from './Home.module.css';

export const Home: FC = () => {
  const {
    status,
    data,
    error,
    // isFetching,
    sendRequest
  } = useApi(getReportList);

  useEffect(() => {
    sendRequest();
  }, [])

  useEffect(() => {
    if (status === 'success' && data) {
      console.log(data);
    } else if (status === 'error' && error) {
      console.log(error);
    }

  }, [status, data, error]);

  return (
    <div className={styles['home-page']}>
      <h1 className={styles['home-page__title']}>
        ИИ-сервис для выявления компьютерных томографий органов грудной клетки с «нормой»
      </h1>
      <div className={styles['home-page__content-grid']}>
      </div>
    </div>
  );
};
