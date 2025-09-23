import type { FC, HTMLAttributes, ReactNode } from 'react';
import styles from './Card.module.css';

interface Props extends HTMLAttributes<HTMLDivElement> {
  className?: string;
  children?: ReactNode;
};

export const Card: FC<Props> = ({ className, children }) => {
  return (
    <div className={
      className
        ? `${styles['card']} ${className}`
        : styles['card']
    }>
      {children}
    </div >
  );
};
