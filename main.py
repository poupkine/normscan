# main.py
import os
import time
from pathlib import Path
import numpy as np
import pydicom
import logging
import gc
import torch
from src.dataloader.dicom_loader import load_dicom_from_zip
from src.model.inference import PathologyDetector
from src.utils.dicom_utils import extract_study_series_uids, extract_file_path_from_dataset
from src.reporter.excel_reporter import generate_excel_report

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "models/autoencoder_2d.pth"
INPUT_DIR = "data/input_zips"
OUTPUT_PATH = "data/output/report.xlsx"


def process_single_dicom_file(ds: pydicom.Dataset, detector: PathologyDetector, zip_path: str) -> dict:
    start_time = time.time()
    try:
        logger.debug(f"--- Начало обработки файла ---")
        # --- Извлечение данных из DICOM ---
        if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
            raise ValueError("Нет пиксельных данных в DICOM файле")

        original_pixel_array = ds.pixel_array
        logger.debug(f"📸 Оригинальная форма pixel_array: {original_pixel_array.shape}")

        # Обработка случая, если pixel_array 3D (многосрезовый DICOM)
        if original_pixel_array.ndim == 3:
            logger.debug("🔄 Обнаружен 3D массив, выбирается первый срез [0]")
            pixel_array_2d = original_pixel_array[0]
        elif original_pixel_array.ndim == 2:
            pixel_array_2d = original_pixel_array
        else:
            raise ValueError(f"Неподдерживаемая размерность pixel_array: {original_pixel_array.ndim}")

        logger.debug(f"📏 Форма 2D массива после обработки: {pixel_array_2d.shape}")
        # Логируем статистику оригинального изображения для отладки
        logger.debug(f"📊 Pixel stats (orig): min={pixel_array_2d.min()}, max={pixel_array_2d.max()}, mean={pixel_array_2d.mean():.2f}")
        pixel_array_float = pixel_array_2d.astype(np.float32)

        # --- Преобразование в HU ---
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        logger.debug(f"📐 RescaleSlope: {slope}, RescaleIntercept: {intercept}")
        hu_image = pixel_array_float * slope + intercept

        # --- Ограничение диапазона HU ---
        hu_image_clipped = np.clip(hu_image, -1000, 400)
        logger.debug(f"📉 HU stats (clipped): min={hu_image_clipped.min()}, max={hu_image_clipped.max()}")

        # --- Нормализация в [0, 1] ---
        min_val, max_val = hu_image_clipped.min(), hu_image_clipped.max()
        if max_val == min_val:
            logger.warning("⚠️ Все значения пикселей одинаковы. Создается нулевой массив.")
            hu_image_norm = np.zeros_like(hu_image_clipped)
        else:
            hu_image_norm = (hu_image_clipped - min_val) / (max_val - min_val)
        logger.debug(f"📈 Нормализованные stats: min={hu_image_norm.min():.4f}, max={hu_image_norm.max():.4f}, mean={hu_image_norm.mean():.4f}")

        # --- Ресемплинг до 128x128 ---
        from scipy.ndimage import zoom
        h, w = hu_image_norm.shape
        if h == 0 or w == 0:
            raise ValueError("Размеры изображения некорректны (0)")
        zoom_factors = (128 / h, 128 / w)
        logger.debug(f"🔎 Исходный размер: ({h}, {w}), Zoom factors: {zoom_factors}")
        slice_resized = zoom(hu_image_norm, zoom_factors, order=1, prefilter=False)
        logger.debug(f"🔎 Размер после ресемплинга: {slice_resized.shape}")
        logger.debug(f"🔎 Ресемплированные stats: min={slice_resized.min():.4f}, max={slice_resized.max():.4f}, mean={slice_resized.mean():.4f}")

        # --- Формируем 4D массив для модели ---
        volume = slice_resized[np.newaxis, np.newaxis, :, :]  # (1, 1, 128, 128)
        logger.debug(f"📦 Форма итогового volume для модели: {volume.shape}")

        # --- Инференс ---
        prob = detector.predict(volume)
        # Округляем вероятность до 4 знаков после запятой
        prob_rounded = round(prob, 4)
        logger.debug(f"🧮 Предсказанная вероятность патологии (округлена): {prob_rounded:.4f}")

        # --- Извлечение метаданных ---
        uid_info = extract_study_series_uids(ds)
        path_to_study = extract_file_path_from_dataset(ds, zip_path)
        logger.debug(f"📁 Path: {path_to_study}")
        logger.debug(f"🆔 Study UID: {uid_info.get('study_uid', '')}")
        logger.debug(f"🔢 Series UID: {uid_info.get('series_uid', '')}")
        logger.debug(f"--- Конец обработки файла ---")

        processing_time = round(time.time() - start_time, 2)
        return {
            'path_to_study': path_to_study,
            'study_uid': uid_info.get('study_uid', ''),
            'series_uid': uid_info.get('series_uid', ''),
            'probability_of_pathology': prob_rounded,  # Используем округленное значение
            'pathology': int(prob_rounded > 0.5),
            'processing_status': 'Success',
            'time_of_processing': processing_time,
            'most_dangerous_pathology_type': None,
            'pathology_localization': None
        }

    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        error_file_name = getattr(ds, 'filename', 'unknown.dcm') if 'ds' in locals() else 'unknown.dcm'
        error_path = f"{zip_path}/{error_file_name}"
        logger.error(f"💀 Ошибка при обработке DICOM-файла {error_path}: {str(e)}")
        return {
            'path_to_study': error_path,
            'study_uid': '',
            'series_uid': '',
            'probability_of_pathology': 0.0,
            'pathology': 0,
            'processing_status': 'Failure',
            'time_of_processing': processing_time,
            'most_dangerous_pathology_type': None,
            'pathology_localization': None
        }


def main(input_dir: str, output_path: str, model_path: str):
    logger.info("🚀 Запуск основного процесса...")
    try:
        logger.info(f"🧠 Инициализация модели из {model_path}")
        detector = PathologyDetector(model_path)
        logger.info("✅ Модель инициализирована")
    except Exception as e:
        logger.error(f"💥 Не удалось инициализировать модель из {model_path}: {e}")
        return

    results = []
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"📂 Директория {input_dir} не существует")
        return

    zip_files = list(input_path.glob("*.zip"))
    logger.info(f"📂 Обнаружено {len(zip_files)} ZIP-файлов.")

    for zip_file in zip_files:
        try:
            logger.info(f"📦 Обработка ZIP-файла: {zip_file.name}")
            dicom_list = load_dicom_from_zip(str(zip_file))
            logger.info(f"📄 В ZIP-файле {zip_file.name} найдено {len(dicom_list)} DICOM-файлов.")

            for i, ds in enumerate(dicom_list):
                # Для ускорения отладки можно ограничить количество обрабатываемых файлов
                # if i >= 10: break
                logger.info(f"  🔧 Обработка файла {i + 1}/{len(dicom_list)}")
                result = process_single_dicom_file(ds, detector, str(zip_file))
                results.append(result)

        except Exception as e:
            logger.error(f"🧨 Критическая ошибка при обработке ZIP-файла {zip_file}: {e}")

    logger.info(f"📊 Всего обработано записей: {len(results)}")
    try:
        logger.info(f"💾 Генерация отчета в {output_path}")
        generate_excel_report(results, output_path)
        logger.info("🏁 Отчет успешно сохранен.")
    except Exception as e:
        logger.error(f"💣 Ошибка при генерации отчета: {e}")


if __name__ == "__main__":
    # Для более подробного логирования во время отладки можно временно поставить DEBUG
    # logging.getLogger().setLevel(logging.DEBUG)
    main(INPUT_DIR, OUTPUT_PATH, MODEL_PATH)
