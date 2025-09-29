# backend/app/routers/predict.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.services.ml_service import MLService
from app.schemas.schemas import PredictionResult, BatchPredictionResult  # ✅ ИСПРАВЛЕНО: ДОБАВИЛИ PredictionResult
from app.utils.excel_reporter import generate_excel_report
import os
import tempfile
import time
from pathlib import Path

router = APIRouter(prefix="/api", tags=["predict"])

# Инициализация ML-сервиса
MODEL_PATH = "app/static/model_weights/autoencoder_2d.pth"
ml_service = MLService(MODEL_PATH)


@router.post("/predict", response_model=PredictionResult)  # ✅ ТЕПЕРЬ РАБОТАЕТ
async def predict_single(file: UploadFile = File(...)):
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # Инференс
        result = ml_service.predict_study(tmp_path)

        # Удаляем временный файл
        os.unlink(tmp_path)

        # Генерируем Excel
        excel_path = "data/output/report.xlsx"
        generate_excel_report([result], excel_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_predict", response_model=BatchPredictionResult)
async def batch_predict(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            result = ml_service.predict_study(tmp_path)
            results.append(result)

            # Удаляем временный файл
            os.unlink(tmp_path)

        # Генерируем Excel
        excel_path = "data/output/report.xlsx"
        generate_excel_report(results, excel_path)

        return BatchPredictionResult(
            results=results,
            excel_file_path=excel_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download_report")
async def download_report():
    """Скачивание последнего сгенерированного отчёта"""
    excel_path = "data/output/report.xlsx"
    if not os.path.exists(excel_path):
        raise HTTPException(status_code=404, detail="Отчёт не найден")
    return FileResponse(excel_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="report.xlsx")
