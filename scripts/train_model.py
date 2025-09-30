# train_model.py
from backend.app.models.autoencoder_2d import Autoencoder2D
from backend.app.services.dicom_loader import load_slices_from_zip
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

# Добавляем backend в путь, чтобы импортировать модули
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "backend"))


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Пути
DATA_DIR = project_root / "backend" / "data" / "train"
MODEL_SAVE_PATH = project_root / "backend" / "app" / "static" / "model_weights" / "autoencoder_2d.pth"
METRICS_REPORT_PATH = project_root / "metrics_report.json"
VAL_REPORT_PATH = project_root / "validation_report.xlsx"

# Создаём директории
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_all_slices_from_normal(data_dir: Path, max_files: int = None) -> np.ndarray:
    """Загружает все срезы из ZIP-файлов в папке normal/"""
    normal_dir = data_dir / "normal"
    if not normal_dir.exists():
        raise FileNotFoundError(f"Папка {normal_dir} не найдена. Убедитесь, что данные лежат в backend/data/train/normal/")

    all_slices = []
    zip_files = list(normal_dir.glob("*.zip"))
    if max_files:
        zip_files = zip_files[:max_files]

    logger.info(f"Найдено {len(zip_files)} ZIP-файлов в {normal_dir}")
    for i, zip_path in enumerate(zip_files):
        logger.info(f"[{i + 1}/{len(zip_files)}] Загрузка {zip_path.name}...")
        try:
            slices, _, _ = load_slices_from_zip(str(zip_path))
            all_slices.append(slices)
        except Exception as e:
            logger.warning(f"Пропущен файл {zip_path}: {e}")

    if not all_slices:
        raise ValueError("Не удалось загрузить ни одного среза!")

    all_slices = np.concatenate(all_slices, axis=0)
    logger.info(f"Всего загружено {len(all_slices)} срезов")
    return all_slices


def train_autoencoder(slices: np.ndarray, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3):
    """Обучает автоэнкодер и возвращает модель + лог потерь"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    # Подготовка данных
    tensor_slices = torch.tensor(slices, dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
    dataset = TensorDataset(tensor_slices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Модель
    model = Autoencoder2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Обучение
    losses = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    training_time_sec = time.time() - start_time
    logger.info("Обучение завершено.")
    return model, losses, training_time_sec, device


def evaluate_on_val_set(model, device, val_dir: Path = None):
    """Оценивает модель на валидации (normal + pathology), возвращает метрики."""
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    from backend.utils.excel_reporter import generate_excel_report

    if val_dir is None:
        val_dir = project_root / "backend" / "data" / "val"
        if not val_dir.exists():
            logger.warning("Папка валидации не найдена. Пропуск оценки качества.")
            return None, None

    normal_dir = val_dir / "normal"
    patho_dir = val_dir / "pathology"

    results = []
    y_true = []
    y_pred = []

    for label, class_dir in [(0, normal_dir), (1, patho_dir)]:
        if not class_dir.exists():
            continue
        for zip_path in class_dir.glob("*.zip"):
            try:
                slices, study_uid, series_uid = load_slices_from_zip(str(zip_path))
                tensor_slices = torch.tensor(slices, dtype=torch.float32).unsqueeze(1).to(device)
                with torch.no_grad():
                    recon = model(tensor_slices)
                    mse = torch.mean((tensor_slices - recon) ** 2, dim=[1, 2, 3])
                    avg_mse = mse.mean().item()
                # Простая сигмоидная калибровка (как в inference)
                prob = 1.0 / (1.0 + np.exp(-100 * (avg_mse - 0.001)))
                pred = 1 if prob > 0.5 else 0

                results.append({
                    "path_to_study": str(zip_path),
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                    "probability_of_pathology": float(prob),
                    "pathology": int(pred),
                    "processing_status": "Success",
                    "time_of_processing": 0.0
                })
                y_true.append(label)
                y_pred.append(prob)
            except Exception as e:
                logger.error(f"Ошибка при обработке {zip_path}: {e}")
                results.append({
                    "path_to_study": str(zip_path),
                    "study_uid": None,
                    "series_uid": None,
                    "probability_of_pathology": None,
                    "pathology": None,
                    "processing_status": "Failure",
                    "time_of_processing": 0.0
                })

    if not y_true:
        return None, results

    auc = roc_auc_score(y_true, y_pred)
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()

    metrics = {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "True_Negative": int(tn),
        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "True_Positive": int(tp),
        "Threshold_used": 0.5
    }

    # Сохраняем Excel-отчёт
    generate_excel_report(results, str(VAL_REPORT_PATH))
    logger.info(f"Валидационный отчёт сохранён: {VAL_REPORT_PATH}")

    return metrics, results


def main():
    logger.info("🚀 Запуск обучения модели автоэнкодера на 'норме'")

    # 1. Загрузка данных
    slices = load_all_slices_from_normal(DATA_DIR)

    # 2. Обучение
    model, losses, train_time, device = train_autoencoder(slices, epochs=30, batch_size=64)

    # 3. Сохранение модели
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"✅ Модель сохранена: {MODEL_SAVE_PATH}")

    # 4. Оценка на валидации
    metrics, _ = evaluate_on_val_set(model, device)

    # 5. Формирование финального отчёта
    report = {
        "model_path": str(MODEL_SAVE_PATH.relative_to(project_root)),
        "training_time_seconds": train_time,
        "num_slices_trained": len(slices),
        "final_loss": losses[-1],
        "epochs": len(losses),
        "device": str(device),
        "metrics_on_validation": metrics or "Validation data not found"
    }

    with open(METRICS_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logger.info(f"📊 Отчёт о метриках сохранён: {METRICS_REPORT_PATH}")
    logger.info("✅ Обучение завершено!")


if __name__ == "__main__":
    main()
