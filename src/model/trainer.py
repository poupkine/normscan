# src/model/trainer.py
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
import pydicom  # Импортируем здесь, так как используем в __main__
from src.dataloader.dicom_loader import load_dicom_from_zip
from src.model.autoencoder_2d import UNet2D

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаём папку models/, если её нет
os.makedirs(os.path.dirname("models/autoencoder_2d.pth"), exist_ok=True)

# --- ОПТИМИЗИРОВАННЫЕ НАСТРОЙКИ ДЛЯ ВАШЕЙ СИСТЕМЫ ---
# Определяем устройство: CUDA (NVIDIA GPU) если доступна, иначе CPU
# Для Windows 11 и 3090 Ti это будет 'cuda'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Используемое устройство для обучения: {DEVICE}")

# Гиперпараметры обучения
# BATCH_SIZE: Увеличиваем для использования памяти 3090 Ti (24 ГБ)
# 64-128 - хороший диапазон для 2D U-Net на 3090
BATCH_SIZE = 128

# NUM_EPOCHS: Количество эпох. 100-200 типично для хорошей сходимости.
NUM_EPOCHS = 150

# LEARNING_RATE: Стандартная скорость обучения для Adam
LEARNING_RATE = 1e-3

# NUM_WORKERS: Количество подпроцессов для загрузки данных.
# Для Windows часто стабильнее 0 или 1, но 4 может ускорить, если HDD быстрый.
# Попробуем 4, если будут проблемы с DataLoader, уменьшить до 0.
NUM_WORKERS = 4

# PIN_MEMORY: Ускоряет перенос данных CPU->GPU, если True.
# Очень полезно для NVIDIA GPU.
PIN_MEMORY = True if DEVICE.type == 'cuda' else False

# Путь для сохранения модели
SAVE_PATH = "models/autoencoder_2d.pth"


class CTNormalDataset(Dataset):
    """Набор данных, загружающий и обрабатывающий каждый DICOM-срез отдельно."""

    def __init__(self, zip_files):
        self.zip_files = zip_files
        self.all_slices = []  # Список всех (zip_path, ds) кортежей

        logger.info("Начало сбора всех DICOM-срезов из ZIP-файлов...")
        for zip_path in self.zip_files:
            # Загружаем список pydicom.Dataset из каждого ZIP
            dicom_list = load_dicom_from_zip(str(zip_path))
            for ds in dicom_list:
                # Проверяем, есть ли пиксельные данные
                if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                    # Добавляем кортеж (путь к ZIP, объект Dataset)
                    self.all_slices.append((str(zip_path), ds))
            logger.info(f"Обработан {zip_path.name}: найдено {len(dicom_list)} DICOM-файлов.")

        logger.info(f"✅ Всего собрано {len(self.all_slices)} DICOM-срезов для обучения.")

    def __len__(self):
        """Возвращает общее количество срезов."""
        return len(self.all_slices)

    def __getitem__(self, idx):
        """
        Загружает и предобрабатывает один DICOM-срез по индексу.
        Возвращает тензор формы (1, 128, 128) типа torch.float32.
        """
        try:
            # Получаем путь к ZIP и объект Dataset
            zip_path, ds = self.all_slices[idx]

            # 1. Извлечение пиксельного массива
            pixel_array = ds.pixel_array.astype(np.float32)

            # 2. Преобразование в HU (Hounsfield Units)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            hu_image = pixel_array * slope + intercept

            # 3. Ограничение диапазона HU (типично для легких)
            hu_image = np.clip(hu_image, -1000, 400)

            # 4. Нормализация в диапазон [0, 1]
            min_val = hu_image.min()
            max_val = hu_image.max()
            if max_val > min_val:
                hu_image_norm = (hu_image - min_val) / (max_val - min_val)
            else:
                hu_image_norm = np.zeros_like(hu_image)

            # 5. Ресемплинг до фиксированного размера (128x128)
            # Используем scipy.ndimage.zoom для точного ресемплинга
            from scipy.ndimage import zoom
            h, w = hu_image_norm.shape
            zoom_factors = (128 / h, 128 / w)
            # order=1 - билинейная интерполяция, хороша для медицинских изображений
            slice_resized = zoom(hu_image_norm, zoom_factors, order=1)

            # 6. Добавление канала: (128, 128) -> (1, 128, 128)
            # DataLoader сам объединит в батч (B, 1, 128, 128)
            tensor = torch.from_numpy(slice_resized).unsqueeze(0).float()

            # --- АУГМЕНТАЦИИ (простые, для регуляризации) ---
            # Случайный шум
            if random.random() > 0.7:  # 30% вероятность
                noise = torch.randn_like(tensor) * 0.02  # Небольшой шум
                tensor = tensor + noise
                tensor = torch.clamp(tensor, 0.0, 1.0)

            # Случайный сдвиг (простая реализация)
            # if random.random() > 0.5:
            #     shift = random.randint(-5, 5)
            #     tensor = torch.roll(tensor, shifts=shift, dims=1)
            #     tensor = torch.roll(tensor, shifts=shift, dims=2)

            return tensor

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке среза idx={idx} из {getattr(ds, 'filename', 'Unknown')} в {zip_path}: {e}")
            # Возвращаем нулевой тензор в случае ошибки, DataLoader может обработать
            return torch.zeros((1, 128, 128), dtype=torch.float32)


def train_autoencoder():
    """Основная функция для обучения автокодировщика."""
    logger.info("========== ЗАПУСК ОБУЧЕНИЯ ==========")

    # 1. Поиск ZIP-файлов с обучающими данными
    train_data_path = Path("data/train_normal")
    zip_files = list(train_data_path.glob("*.zip"))
    logger.info(f"Найдено {len(zip_files)} ZIP-файлов в {train_data_path}.")

    if len(zip_files) == 0:
        logger.error(f"❗ В папке {train_data_path} не найдено ZIP-файлов. Поместите туда ZIP-архивы с нормальными КТ.")
        raise FileNotFoundError(f"Нет данных в {train_data_path}")

    # 2. Создание объекта набора данных
    logger.info("Инициализация CTNormalDataset...")
    dataset = CTNormalDataset(zip_files)

    if len(dataset) == 0:
        logger.error("❗ Набор данных пуст. Проверьте содержимое ZIP-файлов.")
        raise ValueError("Набор данных пуст.")

    # 3. Создание DataLoader
    # - batch_size: Размер батча
    # - shuffle: Перемешивать данные в каждой эпохе
    # - num_workers: Подпроцессы для загрузки (может ускорить на быстрых дисках)
    # - pin_memory: Ускоряет перенос на GPU
    # - persistent_workers: (для PyTorch >= 1.7) Повторное использование воркеров
    logger.info("Создание DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    logger.info(f"DataLoader создан. Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}, Pin Memory: {PIN_MEMORY}")

    # 4. Инициализация модели, функции потерь и оптимизатора
    logger.info("Инициализация модели UNet2D...")
    # in_channels=1 (градации серого), out_channels=1 (восстановленное изображение)
    model = UNet2D(in_channels=1, out_channels=1, base_features=32).to(DEVICE)

    # MSE loss - стандарт для автокодировщиков
    criterion = nn.MSELoss()

    # Adam optimizer - эффективен и адаптивен
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info(f"Модель, функция потерь и оптимизатор инициализированы. LR: {LEARNING_RATE}")

    # 5. Цикл обучения
    logger.info(f"🚀 Начало обучения на {DEVICE} на {NUM_EPOCHS} эпохах...")
    for epoch in range(NUM_EPOCHS):
        model.train()  # Переводим модель в режим обучения
        total_loss = 0.0
        # Используем tqdm для визуализации прогресса
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            # batch - тензор формы (B, 1, 128, 128)
            try:
                # 5.1. Переносим батч на устройство (GPU/CPU)
                batch = batch.to(DEVICE, non_blocking=PIN_MEMORY)

                # 5.2. Обнуляем градиенты
                optimizer.zero_grad()

                # 5.3. Прямой проход (предсказание)
                recon_batch = model(batch)

                # 5.4. Вычисление функции потерь
                loss = criterion(recon_batch, batch)  # MSE между оригиналом и восстановлением

                # 5.5. Обратный проход (вычисление градиентов)
                loss.backward()

                # 5.6. Шаг оптимизатора (обновление весов)
                optimizer.step()

                # 5.7. Накапливаем loss для статистики
                total_loss += loss.item()

                # Обновляем tqdm bar с текущим loss
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            except RuntimeError as re:
                if "out of memory" in str(re).lower():
                    logger.error(f"⚠️  CUDA out of memory на батче {batch_idx}. Пропускаем батч.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.error(f"⚠️  RuntimeError на батче {batch_idx}: {re}")
                # Продолжаем обучение
                continue
            except Exception as e:
                logger.error(f"⚠️  Неожиданная ошибка на батче {batch_idx}: {e}")
                continue

        # 6. Логирование в конце каждой эпохи
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Средний Loss: {avg_loss:.6f}")

        # 7. Сохранение модели после каждой эпохи (можно оптимизировать, сохраняя лучшую)
        try:
            torch.save(model.state_dict(), SAVE_PATH)
            logger.info(f"✅ Модель сохранена в {SAVE_PATH}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении модели: {e}")

    logger.info("🎉 Обучение завершено!")
    logger.info(f"Финальная модель сохранена в {SAVE_PATH}")


if __name__ == "__main__":
    train_autoencoder()
