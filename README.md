<<<<<<< HEAD
# 🩺 NORMSCAN — ИИ-сервис для выявления КТ ОГК с «нормой»

Сервис автоматически анализирует ZIP-архивы с DICOM-снимками компьютерной томографии органов грудной клетки и определяет, содержит ли исследование признаки патологии.  
Реализован **альтернативный подход**: модель обучена на «норме», и всё, что от неё отклоняется, классифицируется как патология.

---
=======
### 🩺 NORMSCAN — ИИ-сервис для выявления КТ ОГК с «нормой»
Сервис анализирует ZIP-архивы с DICOM-снимками компьютерной томографии органов грудной клетки и автоматически определяет, содержит ли исследование признаки патологии.
Реализован альтернативный подход: модель обучена на «норме» и выявляет отклонения как патологию.
>>>>>>> a1e605563221cb81aeb59dd7079a056bbdd783b4

## 🧰 Стек

### Бэкенд
- **Python 3.10**
- **FastAPI 0.115.0**
- **PyTorch 2.3.1 + CUDA 12.1** (поддержка GPU)
- **pydicom 3.0.1**
- **pandas 2.3.2 + openpyxl 3.1.5**

### Фронтенд
- **React 19.1.1**
- **TypeScript**
- **Vite 5**
- **Axios**
- **Redux Toolkit**
- **Tailwind CSS**

### Инфраструктура
- **Docker 24.0+**
- **Docker Compose 2.17.3+**
- **Nginx** (встроенный в образ фронтенда)

---

## 🚀 Установка и запуск

Проект полностью контейнеризован. Для запуска требуется **только Docker и Docker Compose**.

---
### 1. Клонируйте репозиторий

```sh


git clone https://github.com/poupkine/normscan.git
cd normscan
```
## 2. Подготовьте модель
Скопируйте предобученную модель в папку бэкенда:

### Убедитесь, что файл autoencoder_2d.pth существует
```sh
cp /путь/к/вашей/модели/autoencoder_2d.pth backend/app/static/model_weights/
```

## 3. Запустите сервис
```sh
docker compose up --build -d
```

🌐 Доступные ресурсы после запуска
http://localhost:3000 — веб-интерфейс (загрузка ZIP, таблица результатов, скачивание Excel)
http://localhost:8000/docs — Swagger UI (API-документация)
http://localhost:8000/api/predict — эндпоинт для одного исследования
http://localhost:8000/api/batch_predict — эндпоинт для пакетной обработки
http://localhost:8000/api/download_report — скачивание последнего Excel-отчёта

📂 Структура проекта
```sh
normscan/
├── backend/                    # FastAPI + ML-модель
│   ├── app/
│   │   ├── models/             # Autoencoder2D, StudyPathologyDetector
│   │   ├── services/           # dicom_loader.py, ml_service.py
│   │   ├── schemas/            # Pydantic-модели для API
│   │   ├── utils/              # excel_reporter.py
│   │   └── static/model_weights/  # autoencoder_2d.pth
│   ├── main.py                 # Точка входа FastAPI
│   └── Dockerfile
│
├── frontend/                   # React + TypeScript
│   ├── src/
│   │   ├── components/         # UploadForm, ResultsTable, ReportDownloader
│   │   └── services/           # api.ts — вызовы к /api/*
│   └── Dockerfile
│
├── docker-compose.yml          # Основной файл для запуска
├── README.md
└── scripts/
    └── train_autoencoder.py    # Скрипт для обучения модели
```    

* Входные данные: ZIP-архивы с DICOM-файлами (с расширением .dcm или без). Все срезы в архиве должны относиться к одному исследованию.
* Выходные данные: Excel-файл (report.xlsx) с колонками:
    * path_to_study
    * series_uid
    * probability_of_pathology
    * pathology (0 = норма, 1 = патология)
    * time_of_processing

* GPU: Сервис автоматически использует CUDA, если доступна.
* Безопасность: Nginx блокирует WebDAV-методы (PROPFIND), защищая от сканирования.


