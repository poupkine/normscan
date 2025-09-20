# src/model/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model.autoencoder_2d import Autoencoder2D
from src.dataloader.dicom_loader import load_slices_from_zip
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_autoencoder(data_dir: str, model_path: str, epochs=30, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Загружаем только нормальные ZIP
    normal_dir = os.path.join(data_dir, 'normal')
    if not os.path.exists(normal_dir):
        raise FileNotFoundError(f"Папка {normal_dir} не существует")

    all_slices = []
    logger.info(f"Загрузка срезов из {normal_dir}...")

    for zip_file in os.listdir(normal_dir):
        if zip_file.endswith('.zip'):
            zip_path = os.path.join(normal_dir, zip_file)
            slices, _, _ = load_slices_from_zip(zip_path)  # UID не нужен для обучения
            all_slices.append(slices)

    all_slices = np.concatenate(all_slices, axis=0)  # [N, 256, 256]
    logger.info(f"Загружено {len(all_slices)} срезов из {len(os.listdir(normal_dir))} ZIP-файлов")

    # Преобразуем в тензор [N, 1, 256, 256]
    tensor_slices = torch.tensor(all_slices, dtype=torch.float32).unsqueeze(1).to(device)

    dataloader = DataLoader(tensor_slices, batch_size=batch_size, shuffle=True)

    logger.info("Начало обучения...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ Модель сохранена в {model_path}")
