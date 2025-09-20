# backend/app/utils/excel_reporter.py
import pandas as pd
import os
from app.schemas.schemas import PredictionResult


def generate_excel_report(results: List[PredictionResult], output_path: str) -> str:
    """Генерирует Excel-отчёт по требуемой структуре."""
    df_data = []
    for result in results:
        df_data.append({
            "path_to_study": result.path_to_study,
            "study_uid": result.study_uid,
            "series_uid": result.series_uid,
            "probability_of_pathology": result.probability_of_pathology,
            "pathology": result.pathology,
            "processing_status": result.processing_status,
            "time_of_processing": result.time_of_processing,
            "most_dangerous_pathology_type": result.most_dangerous_pathology_type,
            "pathology_localization": result.pathology_localization
        })

    df = pd.DataFrame(df_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path
