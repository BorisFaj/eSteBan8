import torch
from torch import nn
from world.data_handler import PairedImageDataset
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import mlflow
import os
from tqdm import tqdm
from image.discriminator import Discriminator
from world.generator import Generator
from image.mlflow_utils import log_gpu_stats, log_model_histograms, start_mlflow
from torch.utils.data import DataLoader, random_split, Subset
from dotenv import load_dotenv
import math
import torch.nn.functional as F
from ultralytics import YOLO


def should_train_discriminator(
        disc_loss: float,
        gen_loss: float,
        writer,
        epoch,
        disc_loss_target: float,
        sharpness: float,
        gen_weight: float = 0.5
) -> bool:
    """
    Decide si entrenar el discriminador teniendo en cuenta tanto su propia pérdida como la del generador.

    - `disc_loss_target`: objetivo ideal de la pérdida del discriminador.
    - `sharpness`: determina la brusquedad de la transición (mayor → más brusca).
    - `gen_weight`: peso entre 0 y 1 que controla cuánto influye la pérdida del generador.

    Devuelve True si debe entrenarse, usando una probabilidad sigmoide suave.
    """

    # Factor de "bajo rendimiento" del discriminador (0: muy mal, 1: perfecto)
    disc_factor = max(0.0, 1.0 - (disc_loss / disc_loss_target))

    # Factor de "alto esfuerzo" del generador (0: pérdida baja, 1: pérdida alta)
    gen_factor = torch.tanh(torch.tensor(gen_loss)).item()  # normaliza a ~[0, 1]

    # Combinamos: si el discriminador va mal y el generador está sufriendo, mejor no entrenar
    score = gen_weight * gen_factor + (1 - gen_weight) * disc_factor

    # Función sigmoide para suavizar la decisión
    probability = 1 / (1 + math.exp(-sharpness * (score - 0.5)))
    probability = min(max(probability, 0.0), 1.0)

    writer.add_scalar("Debug/Discriminator_Train_Prob", probability, epoch)
    writer.add_scalar("Debug/Discriminator_Score", score, epoch)
    writer.add_scalar("Debug/Discriminator_Factor", disc_factor, epoch)
    writer.add_scalar("Debug/Generator_Factor", gen_factor, epoch)

    return torch.rand(1).item() < probability

def gradient_magnitude(img):
    # Expandimos los filtros Sobel para cada canal (C=3)
    C = img.shape[1]
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).repeat(C, 1, 1, 1)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).repeat(C, 1, 1, 1)

    # Asegúrate de que estén en el mismo dispositivo
    sobel_x = sobel_x.to(img.device)
    sobel_y = sobel_y.to(img.device)

    gx = F.conv2d(img, sobel_x, padding=1, groups=C)
    gy = F.conv2d(img, sobel_y, padding=1, groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2)

def sobel_loss(fake, original):
    grad_fake = gradient_magnitude(fake)
    grad_orig = gradient_magnitude(original)
    return F.l1_loss(grad_fake, grad_orig)

def calc_disc_loss(device, discriminator, real_image, fake_image, criterion):

    fake = torch.zeros((fake_image.size(0), 1), device=device)

    with torch.amp.autocast(device_type="cuda"):
        real_pred = discriminator(real_image)
        fake_pred = discriminator(fake_image.detach())
        valid = torch.ones_like(fake_pred)
        loss_disc = criterion(real_pred, valid) + criterion(fake_pred, fake)

    return loss_disc

def discriminator_step(device, criterion, discriminator, disc_opt, scaler, real_image, fake_image, train):

    disc_loss = calc_disc_loss(device=device, discriminator=discriminator, real_image=real_image,
                               fake_image=fake_image, criterion=criterion)

    if train:
        disc_opt.zero_grad()
        scaler.scale(disc_loss).backward()
        scaler.unscale_(disc_opt)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler.step(disc_opt)
        scaler.update()

    return disc_loss

def log_epoch(writer, epoch, loss_disc, loss_gen):
    writer.add_scalar("Loss/Discriminator", loss_disc.item(), epoch)
    writer.add_scalar("Loss/Generator", loss_gen.item(), epoch)

    mlflow.log_metric("Loss/Discriminator", loss_disc.item(), step=epoch)
    mlflow.log_metric("Loss/Generator", loss_gen.item(), step=epoch)

def save_models(epoch, encoder, discriminator, scaler, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": getattr(encoder, "_orig_mod", encoder).state_dict(),
        "discriminator_state_dict": getattr(discriminator, "_orig_mod", discriminator).state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(checkpoint, path)
    mlflow.log_artifact(path)

    latest_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_latest.pt")
    torch.save(checkpoint, latest_path)
    mlflow.log_artifact(latest_path)
    print(f"✅ Modelos guardados correctamente en {path}")

def evaluate_step(generator, discriminator, test_loader, criterion, writer, device, epoch, face_detector_model):
    generator.eval()
    discriminator.eval()

    total_gen_loss = 0
    total_disc_loss = 0

    with torch.no_grad():
        for step_n, (x, real_img) in enumerate(test_loader):
            x, real_img = x.to(device), real_img.to(device)

            with torch.amp.autocast(device_type="cuda"):
                fake_img = generator(x)
                fake_pred = discriminator(fake_img)

                # Usamos el detector YOLO para obtener las “etiquetas objetivo”
                yolo_targets = yolo_face_score(fake_img, face_detector_model)

                # Evaluamos si el generador está engañando a YOLO
                loss_gen = criterion(fake_pred, yolo_targets)

                # Discriminador sigue con su lógica clásica
                real_pred = discriminator(real_img)
                fake_pred_detach = discriminator(fake_img.detach())
                loss_disc = criterion(real_pred, torch.ones_like(real_pred)) + \
                            criterion(fake_pred_detach, torch.zeros_like(fake_pred_detach))

            total_gen_loss += loss_gen
            total_disc_loss += loss_disc

        avg_adv_loss = total_gen_loss / len(test_loader)
        avg_disc_loss = total_disc_loss / len(test_loader)

        # MLFlow logging por epoch
        mlflow.log_metric("Test/Loss/Discriminator", avg_disc_loss, step=epoch)
        mlflow.log_metric("Test/Loss/Adversarial", avg_adv_loss, step=epoch)

        # TensorBoard logging por epoch
        writer.add_scalar("Test/Loss/Discriminator", avg_disc_loss, epoch)
        writer.add_scalar("Test/Loss/Adversarial", avg_adv_loss, epoch)

        # Imágenes
        real_images_01 = (real_img + 1) / 2
        fake_images_01 = (fake_img + 1) / 2
        img_grid_real = make_grid(real_images_01[:8].cpu().detach(), nrow=4, normalize=True)
        img_grid_fake = make_grid(fake_images_01[:8].cpu(), nrow=4, normalize=True)
        writer.add_image("Test/Images/Real", img_grid_real, epoch)
        writer.add_image("Test/Images/Fake", img_grid_fake, epoch)

        # Debug de diferencias entre fake-images
        img = real_img[:2]  # coge dos imágenes del batch
        _fake = fake_img[:2]

        diff_map = ((_fake[:1] - img[:1]) ** 2).mean(dim=1, keepdim=True)
        norm_diff = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

        writer.add_image("Test/Fake_vs_Real_DiffMap", diff_map[0], epoch)
        writer.add_image("Test/NormalizedDiffMap", norm_diff[0], epoch)

        mask = yolo_targets.squeeze().bool()
        detected_images = fake_images_01[mask][:8]
        if len(detected_images) > 0:
            grid_detected = make_grid(detected_images.cpu(), nrow=4)
            writer.add_image("Test/Detected_by_YOLO", grid_detected, epoch)

    generator.train()
    discriminator.train()

def train_step(device, epoch, generator, discriminator, dataloader, criterion, scaler, opt_disc, opt_gen,
               train_discriminator, face_detector_model):
    generator.train()
    discriminator.train()
    pbar = tqdm(dataloader)
    total_loss_disc = 0
    total_loss_gen = 0
    for step_n, (x, real_img) in enumerate(pbar):
        x, real_img = x.to(device), real_img.to(device)

        with torch.no_grad():
            fake_img_disc = generator(x)

        loss_disc = discriminator_step(
            device=device,
            criterion=criterion,
            discriminator=discriminator,
            disc_opt=opt_disc,
            scaler=scaler,
            real_image=real_img,
            fake_image=fake_img_disc,
            train=train_discriminator)

        with torch.amp.autocast(device_type="cuda"):
            fake_img = generator(x)
            fake_pred = discriminator(fake_img)

            with torch.no_grad():
                yolo_targets = yolo_face_score(fake_img, face_detector_model)  # 1.0 si hay cara, 0.0 si no

            fake_pred = torch.clamp(fake_pred, min=-10, max=10)
            loss_gen = F.binary_cross_entropy_with_logits(fake_pred, yolo_targets)

        print("Loss gen:", loss_gen.item())  # Puede lanzar error si ya es NaN
        print("Loss disc:", loss_disc.item())  # Puede lanzar error si ya es NaN
        print("Fake pred min/max/mean:", fake_pred.min().item(), fake_pred.max().item(), fake_pred.mean().item())
        opt_gen.zero_grad()
        scaler.scale(loss_gen).backward()
        scaler.unscale_(opt_gen)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(opt_gen)
        scaler.update()
        log_gpu_stats(mlflow=mlflow, epoch=epoch)

        pbar.set_description(f"Epoch {epoch} | D: {loss_disc.item():.4f} G: {loss_gen.item():.4f}")
        total_loss_disc += loss_disc
        total_loss_gen += loss_gen

    avg_disc_loss = total_loss_disc / len(dataloader)
    avg_gen_loss = total_loss_gen / len(dataloader)

    return avg_disc_loss, avg_gen_loss, fake_img.detach().cpu()

def train_model(device, start_epoch, num_epochs, scaler, log_dir, generator, discriminator, train_dataset,
                criterion, opt_disc, opt_gen, checkpoint_dir, epochs_to_val, test_loader,
                disc_loss_target, sharpness, warm_up_len, epochs_to_save, batch_size, face_detector_model):

    writer = SummaryWriter(log_dir)
    writer.add_text("Entrenamiento", "Iniciado correctamente", 0)
    writer.flush()

    torch.autograd.set_detect_anomaly(True)

    dataset_size = len(train_dataset)
    samples_per_epoch = dataset_size // num_epochs
    indices = torch.randperm(dataset_size)  # Mezcla aleatoria una vez
    train_discriminator = False

    for epoch in range(start_epoch, num_epochs):
        # Elegimos los índices para esta epoch
        start_idx = epoch * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        subset_indices = indices[start_idx:end_idx]

        # Creamos dataloader con solo esa parte
        train_loader = DataLoader(
            Subset(train_dataset, subset_indices),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

        loss_disc, loss_gen, fake_img = train_step(
            device, epoch, generator, discriminator, train_loader,
            criterion, scaler, opt_disc, opt_gen, train_discriminator, face_detector_model
        )

        if (epoch + 1) % epochs_to_val == 0:
            evaluate_step(generator, discriminator, test_loader, criterion, writer, device, epoch, face_detector_model)
            print("Evaluando sobre el test wey")

        disc_train_next = should_train_discriminator(
            disc_loss=loss_disc,
            gen_loss=loss_gen,
            writer=writer,
            epoch=epoch,
            disc_loss_target=disc_loss_target,
            sharpness=sharpness
        )

        if train_discriminator and not disc_train_next:
            print(f"🧠 [Discriminador]: Paro de entrenar. disc_loss: {loss_disc}")
            train_discriminator = False
        else:
            if epoch > warm_up_len and disc_train_next:
                print("🧠 [Discriminador]: empiezo a entrenar")
                train_discriminator = True

        log_epoch(writer, epoch, loss_disc, loss_gen)

        grid = make_grid((fake_img[:8].detach().cpu() + 1) / 2, nrow=4)
        writer.add_image("Fake", grid, epoch)

        log_model_histograms(writer, generator, "Generator", epoch)
        log_model_histograms(writer, discriminator, "Discriminator", epoch)

        if (epoch + 1) % epochs_to_save == 0:
            save_models(epoch, generator, discriminator, scaler, checkpoint_dir)

        torch.cuda.empty_cache()

    mlflow.end_run()
    writer.close()

def yolo_face_score(img_batch, model):
    scores = []
    for img_tensor in img_batch:
        img_np = ((img_tensor.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype("uint8")
        results = model.predict(img_np, verbose=False)
        detected = len(results[0].boxes) > 0
        scores.append([1.0 if detected else 0.0])
    return torch.tensor(scores, device=img_batch.device, dtype=torch.float32)

def start(device, warm_up_len, num_epochs, epochs_to_val, epochs_to_save, disc_loss_target, sharpness, run_name,
          checkpoint_dir, log_dir, real_img_path, fake_img_path, batch_size, test_split, image_size, image_channels, yolo_face_path):

    dataset = PairedImageDataset(real_img_path, fake_img_path, image_size=image_size)
    print("Tamaño del dataset:", len(dataset))

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    torch.manual_seed(1984)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    yolo_face_detector = YOLO(yolo_face_path).to(device).eval()
    generator = Generator(image_channels=image_channels, image_size=image_size).to(device)
    discriminator = Discriminator(image_channels=image_channels).to(device)

    generator = torch.compile(generator)
    discriminator = torch.compile(discriminator)

    opt_gen = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler()

    # Config MLFlow
    _ = start_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "warm_up_len": warm_up_len,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "EPOCHS_TO_VAL": epochs_to_val,
            "EPOCHS_TO_SAVE": epochs_to_save,
            "disc_loss_target": disc_loss_target,
            "sharpness": sharpness,
            "scheduler": "OneCycleLR",
            "anneal_strategy": "cos"
        },)

        train_model(
            device=device,
            start_epoch=0,
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            discriminator=discriminator,
            generator=generator,
            criterion=criterion,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            scaler=scaler,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            epochs_to_val=epochs_to_val,
            test_loader=test_loader,
            disc_loss_target=disc_loss_target,
            sharpness= sharpness,
            warm_up_len=warm_up_len,
            epochs_to_save=epochs_to_save,
            batch_size=batch_size,
            face_detector_model=yolo_face_detector
        )

if __name__ == "__main__":
    load_dotenv("world/.env")

    # Cuda
    torch.set_float32_matmul_precision('high')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parametros de las redes
    WARM_UP_LEN = int(os.getenv("WARM_UP_LEN"))  # numero de epochs que dejo al discriminador sin entrenar
    BATCH_SIZE = int(os.getenv("batch_size"))
    NUM_EPOCHS = int(os.getenv("num_epochs"))
    IMAGE_LOSS_LAMBDA = float(os.getenv("image_loss_lambda"))  # Parametro para darle algo de tolerancia al image loss
    FREEZE_DISC_LOSS = float(
        os.getenv("FREEZE_DISC_LOSS"))  # loss maximo que alcanza el discriminador antes de ser congelado
    DISC_LOSS_TARGET = float(os.getenv("disc_loss_target"))
    IMAGE_CHANNELS = int(os.getenv("image_channels"))
    IMAGE_SIZE = int(os.getenv("image_size"))
    EPOCHS_TO_VAL = int(os.getenv("EPOCHS_TO_VAL"))  # numero de epochs entre validaciones
    EPOCHS_TO_SAVE = int(os.getenv("EPOCHS_TO_SAVE"))  # numero de epochs para guardar el modelo
    RUN_NAME = os.getenv("RUN_NAME")
    SHARPNESS = float(os.getenv("sharpness"))
    PCT_START = float(os.getenv("pct_start"))
    IMAGE_INPUT_RES = int(os.getenv("IMAGE_INPUT_RES"))
    FAKE_IMG_PATH = os.getenv("fake_img_path")
    REAL_IMG_PATH = os.getenv("real_img_path")
    OPEN_IMG_PATH = os.getenv("open_images_path")
    YOLO_FACE_PATH = os.getenv("yolo_face_path")
    TEST_SPLIT = float(os.getenv("TEST_SPLIT"))

    CHECKPOINT_DIR = os.path.join(os.getenv("checkpoint_dir"), RUN_NAME)
    LOG_DIR = os.path.join(os.getenv("log_dir"), RUN_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start(
        device=DEVICE,
        warm_up_len=WARM_UP_LEN,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        epochs_to_val=EPOCHS_TO_VAL,
        epochs_to_save=EPOCHS_TO_SAVE,
        disc_loss_target=DISC_LOSS_TARGET,
        sharpness=SHARPNESS,
        run_name=RUN_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        real_img_path=REAL_IMG_PATH,
        fake_img_path=FAKE_IMG_PATH,
        test_split=TEST_SPLIT,
        image_size=IMAGE_SIZE,
        image_channels=IMAGE_CHANNELS,
        yolo_face_path=YOLO_FACE_PATH
    )

