import { type FC } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { Container } from './components/Container';

export const Layout: FC = () => {
  return (
    <>
      <Header />
      <main>
        <Container>
          <div>some conent here</div>
        </Container>

      </main>
      <Footer />
    </>
  );
};
