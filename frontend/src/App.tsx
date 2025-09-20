import { Layout } from './layout';
import { Routes, Route } from 'react-router';
import { Home } from './pages/Home';
import './css/style.css';

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path='results' element={<>results</>} />
        <Route path='*' element={<>not found</>} />
      </Route>
    </Routes>
  )
}

export default App;
