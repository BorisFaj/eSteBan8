import os
import torch
import face_recognition
from torch.utils.tensorboard import SummaryWriter
import mlflow
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn
from image.discriminator import Discriminator  # Asumiendo que ya lo tienes definido
from image.mlflow_utils import log_gpu_stats, log_model_histograms, start_mlflow
from dotenv import load_dotenv

# Transforma un tensor a imagen numpy
def tensor_to_numpy_image(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CxHxW → HxWxC
    img = ((img + 1) * 127.5).astype(np.uint8)  # [-1,1] → [0,255]
    return img


# Usa face_recognition para saber si hay una cara
def has_face(img_tensor):
    img_np = tensor_to_numpy_image(img_tensor)
    try:
        faces = face_recognition.face_locations(img_np)
        return 1.0 if len(faces) > 0 else 0.0
    except:
        return 0.0


# Dataset simple con imágenes reales/fake sin labels
class ImageOnlyDataset(Dataset):
    def __init__(self, folder, image_size):
        self.folder = folder
        self.paths = [os.path.join(folder, fname) for fname in os.listdir(folder)
                      if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# Entrenamiento
def train_discriminator(device, dataloader, discriminator, optimizer, num_epochs, log_dir):
    writer = SummaryWriter(log_dir)
    writer.add_text("Entrenamiento", "Iniciado correctamente", 0)
    writer.flush()

    torch.autograd.set_detect_anomaly(True)

    criterion = nn.BCEWithLogitsLoss()
    discriminator.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for img in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img = img.to(device)

            with torch.no_grad():
                labels = torch.tensor([has_face(im) for im in img], device=device).unsqueeze(1)

            preds = discriminator(img)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_gpu_stats(mlflow=mlflow, epoch=epoch)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"🔎 Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/Generator", avg_loss, epoch)
        log_model_histograms(writer, discriminator, "Discriminator", epoch)
        mlflow.log_metric("Loss/Discriminator", avg_loss, step=epoch)

    print("✅ Entrenamiento completado.")
    return discriminator

def save_models(epoch, discriminator, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "discriminator_state_dict": getattr(discriminator, "_orig_mod", discriminator).state_dict(),
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(checkpoint, path)
    mlflow.log_artifact(path)

    latest_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_latest.pt")
    torch.save(checkpoint, latest_path)
    mlflow.log_artifact(latest_path)
    print(f"✅ Modelos guardados correctamente en {path}")

def train_model(run_name, device, num_epochs, batch_size, lr_start, real_img_path, image_size, image_channels,
                epochs_to_val, epochs_to_save, log_dir, checkpoint_dir):

    dataset = ImageOnlyDataset(real_img_path, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    discriminator = Discriminator(image_channels=image_channels).to(device)
    discriminator = torch.compile(discriminator)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_start)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "epochs_to_val": epochs_to_val,
            "epochs_to_save": epochs_to_save,

        }, )

        trained_disc = train_discriminator(
            device=device,
            dataloader=dataloader,
            discriminator=discriminator,
            optimizer=optimizer,
            num_epochs=num_epochs,
            log_dir=log_dir
        )

    # Guardar modelo
    save_models(epoch=num_epochs, discriminator=trained_disc, checkpoint_dir=checkpoint_dir)
    print("💾 Modelo guardado en discriminator_face_trained.pt")


# Configuración principal
if __name__ == "__main__":
    load_dotenv("world/.env")

    # Cuda
    torch.set_float32_matmul_precision('high')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    BATCH_SIZE = int(os.getenv("batch_size"))
    NUM_EPOCHS = int(os.getenv("num_epochs"))
    IMAGE_CHANNELS = int(os.getenv("image_channels"))
    IMAGE_SIZE = int(os.getenv("image_size"))
    EPOCHS_TO_VAL = int(os.getenv("EPOCHS_TO_VAL"))  # numero de epochs entre validaciones
    EPOCHS_TO_SAVE = int(os.getenv("EPOCHS_TO_SAVE"))  # numero de epochs para guardar el modelo
    RUN_NAME = os.getenv("RUN_NAME")
    # LR_START = float(os.getenv("pct_start"))
    LR_START = 1e-4

    REAL_IMG_PATH = os.getenv("real_img_path")
    TEST_SPLIT = float(os.getenv("TEST_SPLIT"))

    CHECKPOINT_DIR = os.path.join(os.getenv("checkpoint_dir"), RUN_NAME)
    LOG_DIR = os.path.join(os.getenv("log_dir"), RUN_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        run_name=RUN_NAME,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr_start=LR_START,
        real_img_path=REAL_IMG_PATH,
        image_size=IMAGE_SIZE,
        epochs_to_val=EPOCHS_TO_VAL,
        epochs_to_save=EPOCHS_TO_SAVE,
        image_channels=IMAGE_CHANNELS,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )
