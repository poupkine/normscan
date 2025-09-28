# backend/app/services/ml_service.py
import torch
import tempfile
import zipfile
import os
from pathlib import Path
import numpy as np
import logging
from ..models.study_pathology_detector import StudyPathologyDetector
from app.models.autoencoder_2d import Autoencoder2D

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.detector = None
        self._init_model()

    def _init_model(self):
        """Инициализация модели с поддержкой CPU/GPU"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"🚀 Используется устройство: {device}")
            self.detector = StudyPathologyDetector(self.model_path, device=device)
            logger.info("✅ Модель успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def predict_from_bytes(self, zip_bytes: bytes, filename: str) -> dict:
        """Обработка ZIP-архива из байтов (для API)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / filename
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)
            result = self.detector.predict_study(str(zip_path))
        return result
