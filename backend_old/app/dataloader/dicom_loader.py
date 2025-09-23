# backend/dataloader/dicom_loader.py
import pydicom
import numpy as np
from pathlib import Path
import zipfile
from typing import List, Tuple


def load_slices_from_zip(zip_path: str, hu_min: int = -1000, hu_max: int = 400) -> Tuple[np.ndarray, str, str]:
    """
    Загружает все DICOM-срезы из ZIP-архива, нормализует их.
    Возвращает: (массив срезов [N, 256, 256], StudyUID, SeriesUID)
    """
    slices = []
    study_uid = "UNKNOWN"
    series_uid = "UNKNOWN"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Получаем список DICOM-файлов
        dicom_files = []
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith('.dcm') or '.' not in file_name:
                try:
                    with zip_ref.open(file_name) as f:
                        ds = pydicom.dcmread(f, stop_before_pixels=True)
                        dicom_files.append((file_name, ds))
                except Exception:
                    # Игнорируем файлы, которые не являются DICOM
                    continue

        if not dicom_files:
            raise ValueError("В архиве не найдено ни одного валидного DICOM-файла.")

        # Сортируем по InstanceNumber
        dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 999999))

        # Получаем UID из первого файла
        first_ds = dicom_files[0][1]
        study_uid = getattr(first_ds, 'StudyInstanceUID', 'UNKNOWN')
        series_uid = getattr(first_ds, 'SeriesInstanceUID', 'UNKNOWN')

        # Загружаем пиксели
        for file_name, ds in dicom_files:
            try:
                with zip_ref.open(file_name) as f:
                    ds_full = pydicom.dcmread(f, stop_before_pixels=False)
                    pixel_array = ds_full.pixel_array.astype(np.float32)

                    # Применяем HU-масштабирование
                    if 'RescaleSlope' in ds_full and 'RescaleIntercept' in ds_full:
                        pixel_array = pixel_array * ds_full.RescaleSlope + ds_full.RescaleIntercept

                    # Нормализуем по заданному HU-диапазону
                    pixel_array = np.clip(pixel_array, hu_min, hu_max)
                    pixel_array = (pixel_array - hu_min) / (hu_max - hu_min)  # [hu_min, hu_max] → [0,1]

                    # Ресайз до 256x256 (если нужно)
                    if pixel_array.shape != (256, 256):
                        # Используем простой resize (для продакшена лучше scipy или skimage)
                        pixel_array = np.resize(pixel_array, (256, 256))

                    slices.append(pixel_array)
            except Exception as e:
                # Пропускаем битые срезы, но логируем (в продакшене можно использовать logging)
                print(f"⚠️  Пропущен срез {file_name}: {e}")
                continue

    if not slices:
        raise ValueError("Не удалось загрузить ни одного валидного среза.")

    return np.stack(slices, axis=0), study_uid, series_uid
