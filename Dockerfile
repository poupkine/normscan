FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Скачиваем предобученную модель (или кладём в репозиторий)
# RUN wget https://example.com/models/autoencoder_3d.pth -O models/autoencoder_3d.pth

CMD ["python", "main.py"]