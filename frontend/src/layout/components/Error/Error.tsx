import { type FC } from 'react';
import { useAppSelector, useAppDispatch } from '@store/hooks';
import { selectError, resetError } from '@store/slices/errorSlice';
import styles from './Error.module.css';


export const Error: FC = () => {
  const error = useAppSelector(selectError);
  const dispatch = useAppDispatch();

  if (!error?.message) {
    return null;
  }

  return (
    <div className={styles['error']}>
      <div className={styles['error-dialog']}>
        <h3 className={styles['error-dialog__title']}>
          {error.message.title}
        </h3>
        <div className={styles['error-dialog__content']}>
          <p className={styles['error-dialog__content-text']}>
            {error.message.text}
          </p>
        </div>
        <div className={styles['error-dialog__actions']}>
          <button
            className={`${styles['error-dialog__btn']} btn btn--error`}
            onClick={() => dispatch(resetError())}
          >
            Ок
          </button>
        </div>
      </div>
    </div>
  );
}