# backend/app/model/study_pathology_detector.py
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import zipfile
import pydicom
from typing import Tuple
import logging

from .autoencoder_2d import Autoencoder2D
from ..dataloader.dicom_loader import load_slices_from_zip

logger = logging.getLogger(__name__)


class StudyPathologyDetector:
    def __init__(self, model_path: str, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 Используется устройство: {self.device}")
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        try:
            model = Autoencoder2D().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("✅ Модель загружена")
            return model
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def predict_study(self, zip_path: str) -> dict:
        result = {
            "filename": Path(zip_path).name,
            "processing_status": "Success",
            "error_detail": None,
            "probability_of_pathology": None,
            "study_uid": "UNKNOWN",
            "series_uid": "UNKNOWN",
            "processing_time_sec": 0.0,
            "path_to_study": zip_path,
            "time_of_processing": 0.0,  # ✅ Добавлено для совместимости с excel_reporter.py
        }
        start_time = time.time()
        try:
            slices, study_uid, series_uid = load_slices_from_zip(zip_path)
            result["study_uid"] = study_uid
            result["series_uid"] = series_uid
            if len(slices) == 0:
                raise ValueError("Не удалось извлечь ни одного валидного DICOM-среза.")

            with torch.no_grad():
                input_tensor = torch.tensor(slices, dtype=torch.float32).unsqueeze(1).to(self.device)
                reconstructed = self.model(input_tensor)
                mse = torch.mean((input_tensor - reconstructed) ** 2, dim=[1, 2, 3]).cpu().numpy()
                avg_mse = float(np.mean(mse))
                probability_of_pathology = min(1.0, max(0.0, avg_mse * 10))

            result["probability_of_pathology"] = round(probability_of_pathology, 4)
            result["pathology"] = 1 if probability_of_pathology > 0.5 else 0
            result["processing_time_sec"] = round(time.time() - start_time, 2)
            result["time_of_processing"] = result["processing_time_sec"]  # ✅ Синхронизируем с processing_time_sec

        except Exception as e:
            result["processing_status"] = "Failure"
            result["error_detail"] = str(e)
            result["processing_time_sec"] = round(time.time() - start_time, 2)
            result["time_of_processing"] = result["processing_time_sec"]  # ✅ Даже при ошибке
            logger.error(f"❌ Ошибка при обработке {zip_path}: {e}")

        logger.info(f"✅ Обработано: {zip_path} | UID: {result['study_uid']} | P(pathology)={result['probability_of_pathology']:.4f}")
        return result
