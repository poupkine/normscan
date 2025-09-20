# src/dataloader/dicom_loader.py
import pydicom
import numpy as np
from zipfile import ZipFile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_slices_from_zip(zip_path: str) -> tuple[np.ndarray, str, str]:
    """
    Извлекает все DICOM-срезы из ZIP, возвращает:
    - массив срезов [N, 256, 256]
    - StudyInstanceUID
    - SeriesInstanceUID
    """
    slices = []
    first_ds = None  # для извлечения UID

    with ZipFile(zip_path, 'r') as zip_ref:
        # Ищем все файлы, которые можно прочитать как DICOM — НЕ по расширению, а по содержимому
        dicom_files = []
        for file_name in zip_ref.namelist():
            try:
                with zip_ref.open(file_name) as f:
                    # Проверяем, что это DICOM — читаем только заголовок
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    dicom_files.append((file_name, ds))
            except Exception:
                continue  # Не DICOM — пропускаем

        if not dicom_files:
            raise ValueError(f"В архиве {zip_path} нет DICOM-файлов")

        # Сортируем по InstanceNumber — чтобы сохранить порядок
        dicom_files.sort(key=lambda x: getattr(x[1], 'InstanceNumber', 999999))

        # Извлекаем UID из первого файла
        first_ds = dicom_files[0][1]
        study_uid = getattr(first_ds, 'StudyInstanceUID', 'UNKNOWN')
        series_uid = getattr(first_ds, 'SeriesInstanceUID', 'UNKNOWN')

        # Извлекаем пиксели всех срезов
        for file_name, ds in dicom_files:
            with zip_ref.open(file_name) as f:
                ds_full = pydicom.dcmread(f, stop_before_pixels=False)
                pixel_array = ds_full.pixel_array.astype(np.float32)

                # Применяем HU-масштабирование
                if 'RescaleSlope' in ds_full and 'RescaleIntercept' in ds_full:
                    pixel_array = pixel_array * ds_full.RescaleSlope + ds_full.RescaleIntercept

                # Нормализуем до [0,1] по HU-диапазону (врачи используют [-1000, 400])
                pixel_array = np.clip(pixel_array, -1000, 400)
                pixel_array = (pixel_array + 1000) / 1400  # [-1000,400] → [0,1]

                # Ресайз до 256x256
                if pixel_array.shape != (256, 256):
                    pixel_array = np.resize(pixel_array, (256, 256))

                slices.append(pixel_array)

    # Возвращаем массив срезов [N, 256, 256] и UID
    return np.stack(slices, axis=0), study_uid, series_uid
