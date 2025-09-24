import { type FC } from 'react';
import { Link } from 'react-router';
import { Container } from '../Container';
import spriteUrl from '@assets/sprite.svg';
import styles from './Header.module.css';

export const Header: FC = () => {
  return (
    <header className={styles['header']}>
      <Container>
        <div className={styles['header__wrapper']}>
          <Link className={styles['header__logo-link']} to='/'>
            <svg className={styles['header__logo-image']} width="319" height="42" aria-hidden="true">
              <use xlinkHref={`${spriteUrl}#icon-logo`}></use>
            </svg>
          </Link>
        </div>
      </Container>
    </header>
  );
};
