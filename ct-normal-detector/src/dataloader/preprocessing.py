import numpy as np
import SimpleITK as sitk
from typing import np.ndarray

def dicom_to_numpy_volume(dicom_datasets: List[pydicom.Dataset]) -> np.ndarray:
    """
    Преобразует список DICOM-срезов в 3D numpy-массив (Z, Y, X), нормализует HU.
    """
    # Получаем размеры
    first_ds = dicom_datasets[0]
    rows, cols = first_ds.Rows, first_ds.Columns
    volume = np.zeros((len(dicom_datasets), rows, cols), dtype=np.float32)

    for i, ds in enumerate(dicom_datasets):
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
            # Применяем формулу преобразования HU: slope * pixel + intercept
            slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
            intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
            hu_image = pixel_array * slope + intercept
            volume[i] = hu_image
        except Exception as e:
            logging.warning(f"Ошибка в срезе {i}: {e}")
            volume[i] = np.zeros((rows, cols))

    # Ограничение HU диапазона (легкие: -1000..+400)
    volume = np.clip(volume, -1000, 400)
    return volume

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Нормализует объем в диапазон [0, 1]"""
    min_val, max_val = volume.min(), volume.max()
    if max_val == min_val:
        return np.zeros_like(volume)
    return (volume - min_val) / (max_val - min_val)