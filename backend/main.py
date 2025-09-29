# backend/main.py
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import logging
import os
from pathlib import Path
from app.services.ml_service import MLService
from app.utils.excel_reporter import generate_excel_report

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NORMSCAN API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # для локальной разработки
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://localhost:8000",        # если фронт через nginx на 80 порту
        "https://normscan.ru",        # ваш продакшен-домен
        # Добавьте другие нужные origins
    ],
    allow_credentials=True,
    allow_methods=["*"],              # или ["GET", "POST", "OPTIONS"]
    allow_headers=["*"],              # или конкретные заголовки
)
# Глобальный сервис (lazy init)
ml_service = None
MODEL_PATH = "app/static/model_weights/autoencoder_2d.pth"
REPORT_PATH = "output/report.xlsx"


def handle_generate_excel_report(results: List[Dict], output_path: str):
    # Генерируем отчёт
    Path(output_path).parent.mkdir(exist_ok=True)
    full_report_path = Path(output_path).resolve()  # ← полный абсолютный путь
    try:
        generate_excel_report(results, output_path)
        logger.info(f"✅ Отчёт УСПЕШНО сохранён по пути: {full_report_path}")
        report_ok = True
    except Exception as e:
        logger.error("❌ Не удалось создать Excel-отчёт")
        # raise HTTPException(status_code=500, detail="Report generation failed")
        logger.error(f"❌ ОШИБКА при сохранении отчёта в {full_report_path}: {e}")
        report_ok = False
    return report_ok


def get_ml_service():
    global ml_service
    if ml_service is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Модель не найдена: {MODEL_PATH}")
            raise RuntimeError("Model weights not found")
        ml_service = MLService(MODEL_PATH)
    return ml_service


@app.get("/health")
async def health_check():
    """Health check для мониторинга"""
    return {"status": "healthy"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Обработка одного ZIP-архива"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files allowed")

    try:
        contents = await file.read()
        result = get_ml_service().predict_from_bytes(contents, file.filename)

        handle_generate_excel_report([result], REPORT_PATH)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Ошибка обработки файла {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Обработка нескольких ZIP-архивов и генерация отчёта"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for file in files:
        if not file.filename.endswith('.zip'):
            continue  # Пропускаем не-zip файлы
        try:
            contents = await file.read()
            result = get_ml_service().predict_from_bytes(contents, file.filename)
            results.append(result)
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file.filename}: {str(e)}")
            results.append({
                "file_name": file.filename,
                "processing_status": "Failure",
                "error_message": str(e),
                "probability_of_pathology": None,
                "study_uid": None,
                "series_uid": None,
                "processing_time_sec": 0,
                "path_to_study": file.filename
            })

    report_ok = handle_generate_excel_report(results, REPORT_PATH)
    return JSONResponse(content={"results": results, "report_available": report_ok})


@app.get("/api/download_report")
async def download_report():
    """Скачивание последнего сгенерированного отчёта"""
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=404, detail="Report not found. Process files first.")
    return FileResponse(
        path=REPORT_PATH,
        filename="normscan_report.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
