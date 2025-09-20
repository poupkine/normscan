# src/utils/dicom_utils.py
import pydicom
from typing import Dict, Tuple


def extract_study_series_uids(ds: pydicom.Dataset) -> Dict[str, str]:
    """
    Извлекает StudyInstanceUID и SeriesInstanceUID из одного DICOM-датасета.
    Возвращает словарь.
    """
    # Получаем UID, если их нет - используем пустую строку
    study_uid = getattr(ds, 'StudyInstanceUID', '')
    series_uid = getattr(ds, 'SeriesInstanceUID', '')

    # Принудительно конвертируем в строку, если это не так
    # Это может помочь, если UID представлены в нестандартном виде
    if not isinstance(study_uid, str):
        study_uid = str(study_uid) if study_uid is not None else ''
    if not isinstance(series_uid, str):
        series_uid = str(series_uid) if series_uid is not None else ''

    return {
        'study_uid': study_uid,
        'series_uid': series_uid
    }

# Если где-то ожидается распаковка кортежа, создадим отдельную функцию


def extract_study_series_uids_tuple(ds: pydicom.Dataset) -> Tuple[str, str]:
    """
    Извлекает StudyInstanceUID и SeriesInstanceUID из одного DICOM-датасета.
    Возвращает кортеж.
    """
    uids = extract_study_series_uids(ds)
    return uids['study_uid'], uids['series_uid']


def extract_file_path_from_dataset(ds: pydicom.Dataset, zip_path: str) -> str:
    """
    Возвращает полный путь к файлу внутри ZIP-архива.
    Пример: data/input_zips/norma_anon.zip/norma_anon/123
    """
    # КРИТИЧНО: используем ds.filename, который мы установили в dicom_loader
    if not hasattr(ds, 'filename') or not ds.filename:
        return f"{zip_path}/unknown.dcm"
    return f"{zip_path}/{ds.filename}"
