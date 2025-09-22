import { type FC } from 'react';
import { InfoCard } from './components/InfoCard';
import { ActionCard } from './components/ActionCard';
import { ResultTable } from './components/ResultTable';
import styles from './Home.module.css';

export const Home: FC = () => {

  return (
    <div className={styles['home-page']}>
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
