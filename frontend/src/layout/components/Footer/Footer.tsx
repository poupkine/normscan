import { type FC } from 'react';
import { Container } from '../Container';
import spriteUrl from '@assets/sprite.svg';
import styles from './Footer.module.css';

export const Footer: FC = () => {
  return (
    <footer className={styles['footer']}>
      <Container>
        <div className={styles['footer__wrapper']}>
          <div className={styles['footer__logo-wrapper']}>
            <svg
              className={styles['footer__logo-icon']}
              width="200"
              height="50"
              aria-hidden="true"
            >
              <use xlinkHref={`${spriteUrl}#icon-lct-logo`}></use>
            </svg>
            <svg
              className={styles['footer__logo-icon']}
              width="200"
              height="50"
              aria-hidden="true"
            >
              <use xlinkHref={`${spriteUrl}#icon-moscow-mayor-logo`}></use>
            </svg>
            <svg
              className={styles['footer__logo-icon']}
              width="200"
              height="50"
              aria-hidden="true"
            >
              <use xlinkHref={`${spriteUrl}#icon-inno-dep-logo`}></use>
            </svg>
          </div>
        </div>
      </Container>
    </footer>
  );
};
