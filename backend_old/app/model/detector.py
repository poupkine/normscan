# backend/model/detector.py
import torch
import torch.nn as nn
import numpy as np
import logging
import os
from pathlib import Path
import time
from typing import Optional, Tuple

from ..dataloader.dicom_loader import load_slices_from_zip

logger = logging.getLogger(__name__)


class StudyPathologyDetector:
    def __init__(self, model_path: str, hu_range: Tuple[int, int] = (-1000, 400)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Используется устройство: {self.device}")

        self.hu_min, self.hu_max = hu_range
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        try:
            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, nn.Module):
                model.to(self.device)
                logger.info(f"✅ Модель загружена на {self.device}")
                return model
            else:
                # Предполагаем, что это state_dict
                from .autoencoder_2d import Autoencoder2D  # Убедитесь, что путь корректен
                model = Autoencoder2D()
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                logger.info(f"✅ State dict модели загружен на {self.device}")
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
            "processing_time_sec": 0.0
        }

        start_time = time.time()

        try:
            slices, study_uid, series_uid = load_slices_from_zip(zip_path, self.hu_min, self.hu_max)
            result["study_uid"] = study_uid
            result["series_uid"] = series_uid

            if len(slices) == 0:
                raise ValueError("Не удалось извлечь ни одного валидного DICOM-среза.")

            # Конвертируем в тензор
            slices_tensor = torch.tensor(slices, dtype=torch.float32).unsqueeze(1).to(self.device)  # [N, 1, 256, 256]

            with torch.no_grad():
                reconstructed = self.model(slices_tensor)
                # Вычисляем MSE для каждого среза
                mse_per_slice = torch.mean((slices_tensor - reconstructed) ** 2, dim=[1, 2, 3]).cpu().numpy()
                # Усредняем по всем срезам
                avg_mse = float(np.mean(mse_per_slice))
                # Простая эвристика: чем выше MSE — тем выше вероятность патологии
                probability_of_pathology = min(max(avg_mse * 10, 0.0), 1.0)  # Нормализуем примерно в [0,1]

            result["probability_of_pathology"] = round(probability_of_pathology, 4)

        except Exception as e:
            result["processing_status"] = "Failure"
            result["error_detail"] = str(e)
            logger.error(f"❌ Ошибка при обработке {zip_path}: {e}")

        result["processing_time_sec"] = round(time.time() - start_time, 2)
        return result
