# src/dataloader/preprocessing.py
import torch
import numpy as np

# Глобальные переменные — устанавливаются при обучении
GLOBAL_MIN = -1000.0
GLOBAL_MAX = 400.0
GLOBAL_RANGE = 1400.0


def set_global_stats(min_val, max_val):
    """Устанавливает глобальные статистики после обучения"""
    global GLOBAL_MIN, GLOBAL_MAX, GLOBAL_RANGE
    GLOBAL_MIN = min_val
    GLOBAL_MAX = max_val
    GLOBAL_RANGE = max_val - min_val
    if GLOBAL_RANGE == 0:
        GLOBAL_RANGE = 1.0


def normalize_slice_global(ds) -> torch.Tensor:
    """
    Нормализует DICOM-срез по глобальному диапазону HU [-1000, 400] → [0, 1].
    Должен совпадать с тем, что использовалось при обучении.
    """
    pixel_array = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    hu_image = pixel_array * slope + intercept
    hu_image = np.clip(hu_image, GLOBAL_MIN, GLOBAL_MAX)
    hu_norm = (hu_image - GLOBAL_MIN) / GLOBAL_RANGE

    from scipy.ndimage import zoom
    h, w = hu_norm.shape
    zoom_factors = (128 / h, 128 / w)
    slice_resized = zoom(hu_norm, zoom_factors, order=1, prefilter=False)

    return torch.tensor(slice_resized, dtype=torch.float32).unsqueeze(0)
