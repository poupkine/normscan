# src/dataloader/dicom_loader.py
import zipfile
import pydicom
import logging
import os
import io
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dicom_from_zip(zip_path: str) -> List[pydicom.Dataset]:
    """
    Извлекает DICOM-файлы из ZIP-архива.
    Поддерживает:
    - Файлы с расширением .dcm
    - Файлы без расширения (имена: 1, 2, 3...)
    - Файлы с сигнатурой DICM
    """
    datasets = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            logger.info(f"В архиве {zip_path} найдено {len(all_files)} файлов/папок.")

            # Ищем потенциальные DICOM-файлы
            candidate_files = [f for f in all_files if not f.endswith('/')]
            logger.info(f"🔍 Найдено {len(candidate_files)} потенциальных DICOM-файлов.")

            processed_count = 0
            failed_count = 0

            for file_name in candidate_files:
                full_file_path = f"{zip_path}/{file_name}"  # Для логирования
                try:
                    with zip_ref.open(file_name) as f:
                        file_bytes = f.read()

                        if len(file_bytes) == 0:
                            logger.debug(f"⏭️ Пропущен пустой файл: {full_file_path}")
                            continue

                        # Проверяем сигнатуру DICM
                        has_dicm_signature = file_bytes[:4] == b'DICM'

                        # Проверяем расширение
                        has_dcm_extension = file_name.lower().endswith('.dcm')

                        # Пробуем читать, если есть сигнатура или расширение .dcm
                        if has_dicm_signature or has_dcm_extension:
                            try:
                                ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
                                # Устанавливаем имя файла в атрибут filename
                                # Это КРИТИЧНО для последующей обработки
                                ds.filename = file_name
                                datasets.append(ds)
                                processed_count += 1
                                logger.debug(f"✅ Успешно загружен DICOM: {full_file_path}")
                            except Exception as read_error:
                                # Если не удалось прочитать с force=True, пробуем без
                                try:
                                    ds = pydicom.dcmread(io.BytesIO(file_bytes))
                                    ds.filename = file_name
                                    datasets.append(ds)
                                    processed_count += 1
                                    logger.debug(f"✅ Успешно загружен DICOM (без force): {full_file_path}")
                                except Exception:
                                    failed_count += 1
                                    logger.warning(f"⚠️ Не удалось прочитать DICOM (force=True и без): {full_file_path} | Ошибка: {str(read_error)[:100]}...")
                        else:
                            # Если нет сигнатуры и расширения, пробуем прочитать как DICOM
                            # Иногда DICOM-файлы не имеют сигнатуры в начале
                            try:
                                ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
                                ds.filename = file_name
                                datasets.append(ds)
                                processed_count += 1
                                logger.debug(f"✅ Успешно загружен DICOM (без сигнатуры/расширения): {full_file_path}")
                            except Exception as e:
                                logger.debug(f"⏭️ Пропущен файл (не DICOM): {full_file_path}")

                except Exception as e:
                    failed_count += 1
                    logger.warning(f"⚠️ Ошибка при чтении файла {full_file_path}: {type(e).__name__}: {str(e)[:100]}...")

            logger.info(f"✅ В архиве {zip_path} успешно загружено {processed_count} DICOM-срезов (из {len(candidate_files)} файлов).")
            if failed_count > 0:
                logger.warning(f"⚠️ Не удалось прочитать {failed_count} файлов.")

    except Exception as e:
        logger.error(f"❌ Ошибка при открытии ZIP-архива {zip_path}: {e}")
        raise

    # Сортировка по InstanceNumber
    def get_instance_number(ds):
        try:
            return int(getattr(ds, 'InstanceNumber', float('inf')))
        except (ValueError, TypeError):
            # Используем имя файла, если InstanceNumber некорректен
            if hasattr(ds, 'filename') and ds.filename:
                basename = os.path.basename(ds.filename)
                name_part = basename.split('.')[0]
                try:
                    return int(name_part)
                except ValueError:
                    pass
            return float('inf')  # В конец списка

    datasets.sort(key=get_instance_number)
    return datasets
