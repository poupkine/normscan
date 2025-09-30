import type { FC, ReactNode, BaseSyntheticEvent } from 'react';
import styles from './Dialog.module.css';

interface Props {
  children: ReactNode;
  onSubmit: (e?: BaseSyntheticEvent) => Promise<void>;
}

export const DialogForm: FC<Props> = ({ children, onSubmit }) => {
  return (
    <form className={styles['dialog__form']} onSubmit={onSubmit} noValidate>
      {children}
    </form>
  );
};