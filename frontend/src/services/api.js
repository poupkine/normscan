// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api'; // Для Docker — будет backend:8000

export default axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10 минут
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});
