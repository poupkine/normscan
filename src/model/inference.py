# src/model/inference.py
import torch
import numpy as np
from .autoencoder_2d import UNet2D
import gc
import logging

logger = logging.getLogger(__name__)


class PathologyDetector:
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet2D(in_channels=1, out_channels=1, base_features=32).to(device)
        logger.info(f"Попытка загрузки модели с устройства {device} из {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            logger.info("✅ Словарь состояния модели успешно загружен.")
        except Exception as e:
            logger.error(f"❌ Не удалось загрузить словарь состояния модели: {e}")
            raise
        self.model.eval()
        logger.info("✅ Модель переведена в режим оценки (eval).")

    def predict(self, volume: np.ndarray) -> float:
        """
        Предсказывает вероятность патологии для одного 2D среза.
        Args:
            volume: np.ndarray с формой (1, 1, 128, 128) - батч из одного канала.
        Returns:
            float: Вероятность патологии (0.0 - 1.0).
        """
        logger.debug(f"🔍 Начало предсказания. Форма входного массива: {volume.shape}")
        torch.cuda.empty_cache()
        gc.collect()

        if volume.shape != (1, 1, 128, 128):
            logger.warning(f"⚠️ Неожиданная форма входного массива: {volume.shape}. Ожидается (1, 1, 128, 128)")

        mse = 1.0  # Значение по умолчанию в случае ошибки
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(volume).float().to(self.device)
                logger.debug(f"🔍 Stats входного тензора: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")

                recon = self.model(tensor)
                logger.debug(f"🔍 Форма тензора восстановления: {recon.shape}")
                logger.debug(f"🔍 Stats восстановленного тензора: min={recon.min().item():.4f}, max={recon.max().item():.4f}, mean={recon.mean().item():.4f}")

                mse = torch.mean((tensor - recon) ** 2).item()
                logger.debug(f"🔍 Вычисленная MSE: {mse:.6f}")

        except Exception as e:
            logger.error(f"❌ Ошибка во время предсказания модели: {e}", exc_info=True)
            # В случае ошибки модели возвращаем высокую вероятность патологии
            mse = 1.0

        finally:
            if 'tensor' in locals():
                del tensor
            if 'recon' in locals():
                del recon
            torch.cuda.empty_cache()
            gc.collect()

        # Нормализуем MSE в вероятность
        # Порог 0.008 должен быть подобран на валидационном наборе.
        # Если MSE всегда большая, вероятность будет всегда 1.0
        # Если MSE близка к 0, вероятность будет близка к 0.
        # Для теста используем более "жесткий" порог, чтобы видеть различия
        # probability = min(mse / 0.008, 1.0)
        # Или можно использовать сигмоиду для более плавного перехода
        # probability = 1.0 / (1.0 + np.exp(-100 * (mse - 0.01))) # Пример сигмоиды

        # Вернемся к линейной нормализации, но с более явным порогом
        # Предположим, что "норма" дает MSE < 0.01, "патология" > 0.02
        # Это нужно подбирать на практике
        BASELINE_MSE = 0.01  # Средняя MSE на нормальных данных
        THRESHOLD_MSE = 0.03  # MSE, выше которой считаем патологией
        if mse <= BASELINE_MSE:
            probability = 0.0
        elif mse >= THRESHOLD_MSE:
            probability = 1.0
        else:
            # Линейная интерполяция между BASELINE и THRESHOLD
            probability = (mse - BASELINE_MSE) / (THRESHOLD_MSE - BASELINE_MSE)

        logger.debug(f"📊 Нормализованная вероятность (MSE={mse:.6f}, baseline={BASELINE_MSE}, threshold={THRESHOLD_MSE}): {probability:.4f}")
        return float(probability)
