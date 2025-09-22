# backend/main.py
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

# CORS для фронтенда (важно: должен быть до всех маршрутов!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный сервис (lazy init)
ml_service = None
MODEL_PATH = "app/static/model_weights/autoencoder_2d.pth"


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

    # Генерируем отчёт
    LAST_REPORT_PATH = "output/report.xlsx"
    Path(LAST_REPORT_PATH).parent.mkdir(exist_ok=True)
    generate_excel_report(results, LAST_REPORT_PATH)

    return JSONResponse(content={"results": results, "report_available": True})


@app.get("/api/download_report")
async def download_report():
    """Скачивание последнего сгенерированного отчёта"""
    LAST_REPORT_PATH = "output/report.xlsx"
    if not os.path.exists(LAST_REPORT_PATH):
        raise HTTPException(status_code=404, detail="Report not found. Process files first.")
    return FileResponse(
        path=LAST_REPORT_PATH,
        filename="normscan_report.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
