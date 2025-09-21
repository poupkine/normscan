# src/reporter/excel_reporter.py
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def generate_excel_report(results: List[Dict], output_path: str):
    """
    Генерирует Excel-отчет из списка результатов.
    Args:
        results (List[Dict]): Список словарей с данными по каждому DICOM-файлу.
        output_path (str): Путь для сохранения .xlsx файла.
    """
    if not results:
        logger.warning("Список результатов пуст. Создается пустой отчет.")
        required_cols = [
            'path_to_study', 'study_uid', 'series_uid',
            'probability_of_pathology', 'pathology',
            'processing_status', 'time_of_processing'
        ]
        optional_cols = ['most_dangerous_pathology_type', 'pathology_localization']
        all_cols = required_cols + optional_cols
        df = pd.DataFrame(columns=all_cols)
    else:
        df = pd.DataFrame(results)

        required_cols = [
            'path_to_study', 'study_uid', 'series_uid',
            'probability_of_pathology', 'pathology',
            'processing_status', 'time_of_processing'
        ]
        optional_cols = ['most_dangerous_pathology_type', 'pathology_localization']

        for col in optional_cols:
            if col not in df.columns:
                df[col] = None

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"В результатах отсутствуют обязательные колонки: {missing_cols}")

        df = df[required_cols + optional_cols]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_excel(output_path, index=False)
        logger.info(f"✅ Отчет успешно сохранен в {output_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении отчета в {output_path}: {e}")
        raise

    print(f"Отчет сохранён: {output_path}")
