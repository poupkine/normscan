import { type FC } from 'react';
import { Outlet } from 'react-router';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { Container } from './components/Container';
import { Error } from './components/Error';

export const Layout: FC = () => {
  return (
    <>
      <Header />
      <main>
        <Container>
          <Outlet />
          <Error />
        </Container>
      </main>
      <Footer />
    </>
  );
};
