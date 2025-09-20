# src/model/inference.py
import torch
import numpy as np
import os
import time
from src.model.autoencoder_2d import Autoencoder2D
from src.dataloader.dicom_loader import load_slices_from_zip
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyPathologyDetector:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🧠 Используется устройство: {self.device}")
        self.model = Autoencoder2D().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info("✅ Модель загружена")

    def predict_study(self, zip_path: str) -> dict:
        start_time = time.time()
        result = {
            "path_to_study": zip_path,
            "study_uid": "UNKNOWN",
            "series_uid": "UNKNOWN",
            "probability_of_pathology": 0.0,
            "pathology": 0,
            "processing_status": "Success",
            "time_of_processing": 0.0
        }

        try:
            # Загружаем срезы и UID за один раз
            slices, study_uid, series_uid = load_slices_from_zip(zip_path)
            result["study_uid"] = study_uid
            result["series_uid"] = series_uid

            # Преобразуем в тензор
            tensor_slices = torch.tensor(slices, dtype=torch.float32).unsqueeze(1).to(self.device)

            # Инференс
            with torch.no_grad():
                reconstructed = self.model(tensor_slices)
                mse_per_slice = torch.mean((tensor_slices - reconstructed) ** 2, dim=[1, 2, 3])
                avg_mse = torch.mean(mse_per_slice).item()

            # Вероятность патологии — чем выше MSE, тем выше вероятность
            # Масштабируем: 0.001 → 0.1, 0.01 → 0.9
            probability_of_pathology = 1.0 / (1.0 + np.exp(-100 * (avg_mse - 0.001)))
            pathology = 1 if probability_of_pathology > 0.5 else 0

            result.update({
                "probability_of_pathology": float(probability_of_pathology),
                "pathology": pathology,
                "time_of_processing": time.time() - start_time
            })

            logger.info(f"✅ Обработано: {zip_path} | UID: {study_uid} | P(pathology)={probability_of_pathology:.4f}")

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {zip_path}: {e}")
            result["processing_status"] = "Failure"
            result["time_of_processing"] = time.time() - start_time

        return result
