# scripts/train_autoencoder.py
import os
from src.model.trainer import train_autoencoder

if __name__ == "__main__":
    DATA_DIR = "data/test_dataset"
    MODEL_PATH = "data/model_weights/autoencoder_2d.pth"

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    train_autoencoder(data_dir=DATA_DIR, model_path=MODEL_PATH, epochs=30)
