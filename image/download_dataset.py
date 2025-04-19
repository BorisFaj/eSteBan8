import fiftyone.zoo as foz
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuración
RESOLUTION = (384, 384)
DATASET_NAME = "open-images-v6"
SAVE_DIR = Path("openimages_custom")
TRAIN_DIR = SAVE_DIR / "train" / "default"
VAL_DIR = SAVE_DIR / "val" / "default"
NUM_TRAIN = 6000
NUM_VAL = 2000

def resize_and_save(src_path, dst_path):
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(RESOLUTION, Image.BICUBIC)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
    except Exception as e:
        print(f"Error con {src_path}: {e}")

def download_and_prepare():
    print("⬇️  Descargando conjunto de entrenamiento...")
    train = foz.load_zoo_dataset(
        DATASET_NAME,
        split="train",
        max_samples=NUM_TRAIN,
        dataset_dir=str(SAVE_DIR / "raw_train"),
        shuffle=True,
        label_types=[],
    )

    print("⬇️  Descargando conjunto de validación...")
    val = foz.load_zoo_dataset(
        DATASET_NAME,
        split="validation",
        max_samples=NUM_VAL,
        dataset_dir=str(SAVE_DIR / "raw_val"),
        shuffle=True,
        label_types=[],
    )

    print("🖼️  Redimensionando entrenamiento...")
    for sample in tqdm(train, desc="Train"):
        resize_and_save(sample.filepath, TRAIN_DIR / Path(sample.filepath).name)

    print("🖼️  Redimensionando validación...")
    for sample in tqdm(val, desc="Val"):
        resize_and_save(sample.filepath, VAL_DIR / Path(sample.filepath).name)

    print("✅ Dataset preparado en", SAVE_DIR)

if __name__ == "__main__":
    download_and_prepare()
