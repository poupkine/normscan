import os
import time
import zipfile
from pathlib import Path
from src.dataloader.dicom_loader import load_dicom_from_zip
from src.dataloader.preprocessing import dicom_to_numpy_volume
from src.model.inference import PathologyDetector
from src.utils.dicom_utils import extract_study_series_uids
from src.reporter.excel_reporter import generate_excel_report
import logging

logging.basicConfig(level=logging.INFO)

def process_single_study(zip_path: str, detector: PathologyDetector) -> dict:
    start_time = time.time()
    try:
        dicom_list = load_dicom_from_zip(zip_path)
        if len(dicom_list) < 5:
            raise ValueError("Слишком мало срезов")

        volume = dicom_to_numpy_volume(dicom_list)
        prob = detector.predict(volume)
        uid_info = extract_study_series_uids(dicom_list)

        result = {
            'path_to_study': zip_path,
            'study_uid': uid_info['study_uid'],
            'series_uid': uid_info['series_uid'],
            'probability_of_pathology': prob,
            'pathology': int(prob > 0.5),  # бинарная классификация
            'processing_status': 'Success',
            'time_of_processing': round(time.time() - start_time, 2),
            'most_dangerous_pathology_type': None,
            'pathology_localization': None
        }
        return result

    except Exception as e:
        logging.error(f"Ошибка при обработке {zip_path}: {e}")
        return {
            'path_to_study': zip_path,
            'study_uid': '',
            'series_uid': '',
            'probability_of_pathology': 0.0,
            'pathology': 0,
            'processing_status': 'Failure',
            'time_of_processing': round(time.time() - start_time, 2),
            'most_dangerous_pathology_type': None,
            'pathology_localization': None
        }

def main(input_dir: str, output_path: str, model_path: str):
    detector = PathologyDetector(model_path)
    results = []

    zip_files = [f for f in Path(input_dir).glob("*.zip")]
    logging.info(f"Обнаружено {len(zip_files)} ZIP-файлов.")

    for zip_file in zip_files:
        result = process_single_study(str(zip_file), detector)
        results.append(result)

    generate_excel_report(results, output_path)

if __name__ == "__main__":
    INPUT_DIR = "data/input_zips"
    OUTPUT_PATH = "data/output/report.xlsx"
    MODEL_PATH = "models/autoencoder_3d.pth"

    main(INPUT_DIR, OUTPUT_PATH, MODEL_PATH)