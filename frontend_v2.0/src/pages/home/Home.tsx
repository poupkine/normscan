import { type FC } from 'react';
import { useAppSelector } from '@store/hooks';
import { selectResultList } from './slice';
import { InfoCard } from './components/InfoCard';
import { ActionCard } from './components/ActionCard';
import { ResultTable } from './components/ResultTable';
import styles from './Home.module.css';

export const Home: FC = () => {
  const resultList = useAppSelector(selectResultList);

  return (
    <div className={styles['home-page']}>
      <h1 className={styles['home-page__title']}>
        Автоматический анализ КТ-исследований
      </h1>
      <div className={styles['home-page__content-grid']}>
        <InfoCard />
        <ActionCard />
      </div>
      {resultList.length > 0 &&
        <>
          <h2 className={styles['home-page__subtitle']}>Результаты</h2>
          <ResultTable resultList={resultList} />
        </>
      }
      {resultList.length > 1 &&
        <a
          className={styles['home-page__report-link']}
          href='/api/download_report'
          target='_blank'
          rel='noopener noreferrer'
        >
          Скачать отчет
        </a>}
    </div>
  );
};
