# scripts/fine_tune.py
import os
import sys
from src.model.inference import StudyPathologyDetector

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Использование: python fine_tune.py <путь_к_папке_с_данными>")
        print("   Пример: python fine_tune.py data/fine_tune_data")
        print("   Папка должна содержать: data/fine_tune_data/normal/ и/или data/fine_tune_data/pathology/")
        sys.exit(1)

    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f"❌ Папка {data_dir} не существует!")
        sys.exit(1)

    detector = StudyPathologyDetector("data/model_weights/autoencoder_3d.pth")
    detector.fine_tune(data_dir=data_dir, epochs=10, batch_size=2)
