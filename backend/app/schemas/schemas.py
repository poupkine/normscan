# backend/app/schemas/schemas.py
from pydantic import BaseModel
from typing import List, Optional


class PredictionResult(BaseModel):  # ✅ ДОЛЖЕН БЫТЬ
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int
    processing_status: str
    time_of_processing: float
    most_dangerous_pathology_type: Optional[str] = None
    pathology_localization: Optional[str] = None


class BatchPredictionResult(BaseModel):  # ✅ ДОЛЖЕН БЫТЬ
    results: List[PredictionResult]
    excel_file_path: str
