# main.py
import os
from pathlib import Path
from src.model.inference import StudyPathologyDetector
from src.reporter.excel_reporter import generate_excel_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIR = "data/input_zips"
OUTPUT_PATH = "data/output/report.xlsx"
MODEL_PATH = "data/model_weights/autoencoder_2d.pth"


def main():
    logger.info("🚀 ЗАПУСК NORMSCAN v2 — Обучение на срезах")
    logger.info(f"🔍 INPUT_DIR: {INPUT_DIR}")
    logger.info(f"📂 MODEL_PATH: {MODEL_PATH}")
    logger.info(f"📝 OUTPUT_PATH: {OUTPUT_PATH}")

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Модель не найдена: {MODEL_PATH}")
        return

    detector = StudyPathologyDetector(MODEL_PATH)
    results = []

    zip_files = [f for f in Path(INPUT_DIR).glob("*.zip")]
    logger.info(f"✅ Обнаружено {len(zip_files)} ZIP-файлов.")

    for zip_file in zip_files:
        result = detector.predict_study(str(zip_file))
        results.append(result)

    generate_excel_report(results, OUTPUT_PATH)
    logger.info(f"✅ Отчёт сохранён: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
