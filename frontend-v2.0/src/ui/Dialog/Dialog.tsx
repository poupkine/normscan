import type { FC, ReactNode } from 'react';
import { Loader } from '../Loader';
import spriteUrl from '@assets/sprite.svg';
import styles from './Dialog.module.css';

interface Props {
  children: ReactNode;
  title: string;
  isOpen: boolean;
  handleClose: () => void;
  isFetching?: boolean;
}

export const Dialog: FC<Props> = ({
  children,
  title,
  isOpen,
  handleClose,
  isFetching,
}) => {
  if (!isOpen) return null;

  return (
    <div className={styles['dialog']}>
      {isFetching && <Loader />}
      <div className={styles['dialog__box']}>
        <button
          className={styles['dialog__btn-close']}
          type="button"
          aria-label="Закрыть окно"
          onClick={handleClose}
        >
          <svg className={styles['dialog__btn-close-icon']}
            width="28"
            height="28"
            aria-hidden="true"
          >
            <use xlinkHref={`${spriteUrl}#icon-cross-circle`}></use>
          </svg>
        </button>
        <h3 className={styles['dialog__title']}>{title}</h3>
        {children}
      </div>
    </div>
  );
};