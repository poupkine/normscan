import type { FC, TableHTMLAttributes } from 'react';
import styles from './Table.module.css';

interface Props extends TableHTMLAttributes<HTMLTableElement> { }

export const Table: FC<Props> = ({ children, className, ...rest }) => {
  return (
    <table className={
      className
        ? `${styles['table']} ${className}`
        : styles['table']
    }
      {...rest}
    >
      {children}
    </table>
  );
};
