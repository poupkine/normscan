import type { FC, ReactNode } from 'react';
import styles from './Container.module.css';

interface Props {
  children: ReactNode;
}

export const Container: FC<Props> = ({ children }) => {
  return (
    <div className={styles['container']}>
      {children}
    </div>
  );
};
