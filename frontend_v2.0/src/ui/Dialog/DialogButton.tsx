import type { FC, ButtonHTMLAttributes } from 'react';
import styles from './Dialog.module.css';

type ButtonVariant = 'primary' | 'error';

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const getClass = (variant: ButtonVariant): string => {
  switch (variant) {
    case 'primary':
      return `btn ${styles['dialog__btn']}`;
    case 'error':
      return `btn ${styles['dialog__btn']} ${styles['dialog__btn--error']}`;
    default:
      return `btn ${styles['dialog__btn']}`;
  }
};

export const DialogButton: FC<Props> = ({
  children,
  variant = 'primary',
  type = "button",
  ...rest
}) => {
  return (
    <button
      className={getClass(variant)}
      type={type}
      {...rest}
    >
      {children}
    </button>
  );
};