# scripts/evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from src.model.inference import StudyPathologyDetector
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, test_dataset_dir: str):
    """
    Оценивает модель на тестовом наборе: normal/ и pathology/
    """
    detector = StudyPathologyDetector(model_path)
    y_true = []
    y_pred = []

    for class_name in ['normal', 'pathology']:
        label = 0 if class_name == 'normal' else 1
        class_dir = Path(test_dataset_dir) / class_name
        for zip_file in class_dir.glob("*.zip"):
            result = detector.predict_study(str(zip_file))
            y_true.append(label)
            y_pred.append(result["probability_of_pathology"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # AUC
    auc = roc_auc_score(y_true, y_pred)
    # Confusion Matrix
    y_pred_bin = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"\n=== РЕЗУЛЬТАТЫ ОЦЕНКИ ===")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"Sensitivity (Recall): {sensitivity:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    logger.info(f"\n{classification_report(y_true, y_pred_bin, target_names=['Normal', 'Pathology'])}")


if __name__ == "__main__":
    evaluate_model(
        model_path="data/model_weights/autoencoder_3d.pth",
        test_dataset_dir="data/test_dataset"
    )
