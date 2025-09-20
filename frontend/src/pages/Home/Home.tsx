import { type FC } from 'react';
import styles from './Home.module.css';

export const Home: FC = () => {
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
