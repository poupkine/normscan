import torch
import numpy as np
from .autoencoder_3d import UNet3D
from ..dataloader.preprocessing import normalize_volume

class PathologyDetector:
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet3D().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def predict(self, volume: np.ndarray) -> float:
        """
        Возвращает вероятность патологии (0.0–1.0) на основе MSE реконструкции.
        """
        volume_norm = normalize_volume(volume)
        tensor = torch.tensor(volume_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)  # [1,1,Z,Y,X]

        with torch.no_grad():
            recon = self.model(tensor)
            mse_loss = torch.mean((tensor - recon) ** 2).item()

        # Нормализуем MSE: порог = средняя ошибка на норме (можно задать через валидацию)
        # Пример: если средняя ошибка на норме = 0.015, то:
        probability = min(mse_loss / 0.015, 1.0)  # ограничение до 1.0
        return float(probability)