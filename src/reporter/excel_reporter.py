import pandas as pd
from pathlib import Path
from typing import List, Dict

def generate_excel_report(results: List[Dict], output_path: str):
    """
    results = [
        {
            'path_to_study': '/input/1.zip',
            'study_uid': '1.2.840...',
            'series_uid': '1.2.840...',
            'probability_of_pathology': 0.87,
            'pathology': 1,
            'processing_status': 'Success',
            'time_of_processing': 125.3,
            'most_dangerous_pathology_type': 'pneumonia',  # опционально
            'pathology_localization': [10, 150, 20, 180, 5, 50]  # опционально
        },
        ...
    ]
    """
    df = pd.DataFrame(results)
    required_cols = [
        'path_to_study', 'study_uid', 'series_uid',
        'probability_of_pathology', 'pathology',
        'processing_status', 'time_of_processing'
    ]
    optional_cols = ['most_dangerous_pathology_type', 'pathology_localization']
    
    # Добавляем опциональные колонки, если они есть
    for col in optional_cols:
        if col not in df.columns:
            df[col] = None

    df = df[required_cols + optional_cols]
    df.to_excel(output_path, index=False)
    print(f"Отчет сохранён: {output_path}")