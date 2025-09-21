import { type FC } from 'react';
import { Card } from '@ui/Card';
import spriteUrl from '@assets/sprite.svg';
import styles from './InfoCard.module.css';

export const InfoCard: FC = () => {
  return (
    <Card className={styles['info-card']}>
      <h2 className={styles['info-card__title']}>Как это работает?</h2>
      <div className={styles['info-card__wrapper']}>
        <div className={styles['info-card__item']}>
          <svg className={styles['info-card__item-icon']} width="100" height="100" aria-hidden="true">
            <use xlinkHref={`${spriteUrl}#icon-upload`}></use>
          </svg>
          <span className={styles['info-card__item-title']}>
            Загрузите КТ
          </span>
        </div>
        <div className={styles['info-card__item']}>
          <svg className={styles['info-card__item-icon']} width="100" height="100" aria-hidden="true">
            <use xlinkHref={`${spriteUrl}#icon-gear`}></use>
          </svg>
          <span className={styles['info-card__item-title']}>
            Система анализирует
          </span>
        </div>
        <div className={styles['info-card__item']}>
          <svg className={styles['info-card__item-icon']} width="100" height="100" aria-hidden="true">
            <use xlinkHref={`${spriteUrl}#icon-clipboard`}></use>
          </svg>
          <span className={styles['info-card__item-title']}>
            Получите результат
          </span>
        </div>
      </div>
    </Card>
  );
};
