import zipfile
import pydicom
import os
from typing import List, Dict
import logging

def load_dicom_from_zip(zip_path: str) -> List[pydicom.Dataset]:
    """
    Извлекает все DICOM-файлы из ZIP-архива и возвращает список объектов pydicom.Dataset.
    Сортирует по InstanceNumber для получения правильной последовательности срезов.
    """
    datasets = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.lower().endswith('.dcm'):
                    with zip_ref.open(file) as f:
                        ds = pydicom.dcmread(f, force=True)
                        datasets.append(ds)
    except Exception as e:
        logging.error(f"Ошибка при чтении ZIP {zip_path}: {e}")
        raise

    # Сортировка по InstanceNumber или ImagePositionPatient
    datasets.sort(key=lambda x: getattr(x, 'InstanceNumber', 0))
    return datasets