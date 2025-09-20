# src/reporter/excel_reporter.py
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def generate_excel_report(results: List[Dict], output_path: str):
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

        # Убедиться, что probability_of_pathology является числом с плавающей точкой
        # и форматируется с 4 знаками после запятой при сохранении в Excel
        # ExcelReporter сам по себе не форматирует, форматирование зависит от Excel
        # Но pandas.to_excel может принять float_format
        # df['probability_of_pathology'] = df['probability_of_pathology'].apply(lambda x: f"{x:.4f}")
        # Лучше оставить как float, а форматирование задать в Excel или при выводе

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Используем float_format для числовых значений
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, float_format="%.4f")
        # Получаем объект workbook и worksheet для форматирования
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']  # Имя листа по умолчанию

        # Формат для чисел с 4 знаками после запятой
        number_format = workbook.add_format({'num_format': '0.0000'})

        # Применяем формат к колонке 'probability_of_pathology'
        # Найдем индекс колонки
        for i, col in enumerate(df.columns):
            if col == 'probability_of_pathology':
                # Форматируем всю колонку (кроме заголовка)
                worksheet.set_column(i, i, 15, number_format)
                break

    print(f"Отчет сохранён: {output_path}")
