# scripts/generate_synthetic_normal.py
import numpy as np
import pydicom
import os
from pathlib import Path
from pydicom.dataset import FileMetaDataset

# Создаем папку для синтетических срезов
synthetic_dir = Path("data/test_dataset/normal/synthetic")
synthetic_dir.mkdir(parents=True, exist_ok=True)

# Генерируем 1000 синтетических срезов
for i in range(1000):
    # Создаем срез размером 512x512
    slice_data = np.random.normal(0, 50, (512, 512)).astype(np.int16)  # Гауссов шум

    # Добавляем типичные структуры:
    # 1. Легкие — черные (HU ≈ -800)
    slice_data[100:400, 100:400] = -800

    # 2. Сердце — серое (HU ≈ 40)
    slice_data[200:300, 250:350] = 40

    # 3. Кости — белые (HU ≈ 300)
    slice_data[50:100, 200:300] = 300
    slice_data[400:450, 200:300] = 300

    # 4. Мягкие ткани — серые (HU ≈ 0)
    slice_data[100:400, 50:100] = 0
    slice_data[100:400, 400:450] = 0

    # Создаем DICOM-файл
    ds = pydicom.Dataset()

    # --- Основные атрибуты DICOM ---
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.InstanceNumber = i + 1
    ds.StudyInstanceUID = "1.2.276.0.7230010.3.1.2.2462171185.19116.1754559949.863"
    ds.SeriesInstanceUID = "1.2.276.0.7230010.3.1.3.2462171185.19116.1754559949.864"
    ds.PixelData = slice_data.tobytes()

    # --- КЛЮЧЕВОЙ ШАГ: Создаем и присваиваем file_meta ---
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage  # Стандарт для изображений
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()  # Уникальный UID для каждого файла
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # ✅ КРИТИЧНО: стандартный формат
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Сохраняем
    filename = synthetic_dir / f"synthetic_{i:04d}.dcm"
    ds.save_as(filename)

print(f"✅ Создано 1000 синтетических DICOM-срезов в {synthetic_dir}")
