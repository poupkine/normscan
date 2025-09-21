import { type FC } from 'react';
import { Card } from '@ui/Card';
import spriteUrl from '@assets/sprite.svg';
import styles from './ActionCard.module.css';

export const ActionCard: FC = () => {
  return (
    <Card className={styles['action-card']}>
      <h2 className='visually-hidden'>Загрузить КТ-исследование</h2>
      <svg className={styles['action-card__icon']} width="300" height="224" aria-hidden="true">
        <use xlinkHref={`${spriteUrl}#icon-cloud-upload-big`}></use>
      </svg>
      <button
        className={`btn btn--secondary-light ${styles['action-card__button']}`}
      >
        Загрузить КТ-исследование
      </button>
      <div className={styles['action-card__info']}>
        <span className={styles['action-card__info-text']}>
          Поддерживаются форматы DICOM, ZIP
        </span>
        <span className={styles['action-card__info-text']}>
          Максимальный размер файла 1ГБ
        </span>
        <span className={styles['action-card__info-text']}>
          Исследование будет автоматизировано
        </span>
      </div>
    </Card>
  );
};
