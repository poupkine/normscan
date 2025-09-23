import os
import SimpleITK as sitk
from typing import List, Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_nifti_volume(nifti_path: str) -> np.ndarray:
    """
    Загружает 3D-объем из файла NIfTI (.nii.gz) и возвращает его как numpy массив.
    Масштабирует интенсивности в диапазон [0, 1] для обучения.
    """
    try:
        # Читаем файл с помощью SimpleITK
        image = sitk.ReadImage(nifti_path)

        # Преобразуем в numpy массив
        volume = sitk.GetArrayFromImage(image)

        # Нормализуем интенсивности в диапазон [0, 1]
        # Для КТ-объемов, если данные не в HU, это просто нормализация по мин/макс
        min_val = volume.min()
        max_val = volume.max()
        if max_val != min_val:
            volume = (volume - min_val) / (max_val - min_val)
        else:
            volume = np.zeros_like(volume, dtype=np.float32)

        logger.debug(f"Загружен объем {nifti_path} размером {volume.shape}")
        return volume.astype(np.float32)

    except Exception as e:
        logger.error(f"Ошибка при загрузке NIfTI файла {nifti_path}: {e}")
        raise


def load_nifti_mask(nifti_mask_path: str) -> np.ndarray:
    """
    Загружает бинарную маску из файла NIfTI (.nii.gz) и возвращает ее как numpy массив.
    """
    try:
        mask = sitk.ReadImage(nifti_mask_path)
        mask_array = sitk.GetArrayFromImage(mask)
        return mask_array.astype(np.uint8)
    except Exception as e:
        logger.error(f"Ошибка при загрузке маски NIfTI {nifti_mask_path}: {e}")
        raise


def get_nifti_files_by_class(base_dir: str, class_name: str) -> List[str]:
    """
    Возвращает список путей к NIfTI-файлам для указанного класса (например, 'CT-0' для нормы).
    Ожидаемая структура: base_dir/studies/CT-0/study_XXX.nii.gz
    """
    class_dir = os.path.join(base_dir, "studies", class_name)
    if not os.path.exists(class_dir):
        raise FileNotFoundError(f"Директория класса {class_name} не найдена: {class_dir}")

    nifti_files = []
    for file_name in os.listdir(class_dir):
        if file_name.lower().endswith('.nii.gz'):
            nifti_files.append(os.path.join(class_dir, file_name))

    logger.info(f"Найдено {len(nifti_files)} NIfTI-файлов в классе {class_name}.")
    return nifti_files


def get_nifti_masks_for_volumes(base_dir: str, volume_paths: List[str]) -> List[str]:
    """
    Сопоставляет список путей к объемам с путями к их маскам.
    Возвращает список путей к маскам в том же порядке.
    """
    masks = []
    for vol_path in volume_paths:
        # Извлекаем имя файла без расширения (например, "study_0256")
        base_name = os.path.basename(vol_path).replace('.nii.gz', '')
        # Формируем имя маски
        mask_name = f"{base_name}_mask.nii.gz"
        mask_path = os.path.join(base_dir, "masks", mask_name)
        if os.path.exists(mask_path):
            masks.append(mask_path)
        else:
            masks.append(None)  # Маска не найдена
            logger.warning(f"Маска для {vol_path} не найдена: {mask_path}")
    return masks
