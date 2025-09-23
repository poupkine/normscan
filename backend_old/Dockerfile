# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt uvicorn

# Копируем ВСЁ приложение
COPY . .

# Создаём директорию для отчётов
RUN mkdir -p /app/output

EXPOSE 8000

# Запускаем без reload в продакшене
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
