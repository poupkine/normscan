import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
import random
from pathlib import Path
from tqdm import tqdm
from src.dataloader.dicom_loader import load_dicom_from_zip
from src.model.autoencoder_2d import UNet2D  # ← НОВАЯ 2D МОДЕЛЬ
import pydicom  # ← ✅ КРИТИЧНО: ИМПОРТ ДЛЯ isinstance(slope, pydicom.valuerep.DSfloat)


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаём папку models/, если её нет — КРИТИЧНО!
os.makedirs(os.path.dirname("models/autoencoder_2d.pth"), exist_ok=True)

# Настройки обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
SAVE_PATH = "models/autoencoder_2d.pth"


class CTNormalDataset(Dataset):
    def __init__(self, zip_files):
        self.zip_files = zip_files
        self.all_slices = []  # Список всех срезов со всех ZIP-файлов

        # Проходим по всем ZIP-файлам и собираем ВСЕ срезы
        for zip_path in zip_files:
            dicom_list = load_dicom_from_zip(zip_path)
            for ds in dicom_list:
                if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                    self.all_slices.append((zip_path, ds))

        logger.info(f"Собрано {len(self.all_slices)} срезов из {len(zip_files)} ZIP-файлов.")

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, idx):
        zip_path, ds = self.all_slices[idx]

        # Получаем пиксельные данные
        pixel_array = ds.pixel_array.astype(np.float32)

        # Преобразование в HU
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        if isinstance(slope, pydicom.valuerep.DSfloat):
            slope = float(slope)
        if isinstance(intercept, pydicom.valuerep.DSfloat):
            intercept = float(intercept)
        hu_image = pixel_array * slope + intercept

        # Ограничение диапазона HU
        hu_image = np.clip(hu_image, -1000, 400)

        # Нормализация в [0, 1]
        min_val, max_val = hu_image.min(), hu_image.max()
        if max_val == min_val:
            slice_norm = np.zeros_like(hu_image)
        else:
            slice_norm = (hu_image - min_val) / (max_val - min_val)

        # Ресемплинг до 128x128
        from scipy.ndimage import zoom
        h, w = slice_norm.shape
        zoom_factors = (128 / h, 128 / w)
        slice_resized = zoom(slice_norm, zoom_factors, order=1, prefilter=False)

        # Добавляем канал: (128, 128) → (1, 128, 128)
        tensor = torch.tensor(slice_resized, dtype=torch.float32).unsqueeze(0)

        # 🔥 АУГМЕНТАЦИИ — обязательны для обобщения
        if random.random() > 0.5:
            noise = torch.randn_like(tensor) * 0.01
            tensor += noise
            tensor = torch.clamp(tensor, 0, 1)

        if random.random() > 0.5:
            shift = random.randint(-8, 8)
            tensor = torch.roll(tensor, shifts=shift, dims=1)
            tensor = torch.roll(tensor, shifts=shift, dims=2)

        return tensor


def train_autoencoder():
    zip_files = list(Path("data/train_normal").glob("*.zip"))
    logger.info(f"Найдено {len(zip_files)} ZIP-файлов с нормой.")

    if len(zip_files) == 0:
        raise FileNotFoundError(
            "Нет данных в data/train_normal/. Поместите ZIP-файлы с нормальными КТ-исследованиями (без патологий)!"
        )

    dataset = CTNormalDataset(zip_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = UNet2D(in_channels=1, out_channels=1, base_features=32).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Начало обучения...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Avg Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), SAVE_PATH)
        logger.info(f"Модель сохранена: {SAVE_PATH}")

    logger.info("✅ Обучение завершено!")


if __name__ == "__main__":
    import pydicom  # ← ИМПОРТ ДЛЯ dicom_loader.py
    train_autoencoder()
