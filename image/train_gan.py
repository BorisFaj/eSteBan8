import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from encoder import Encoder
from discriminator import Discriminator
from torch import amp
import math
from dotenv import load_dotenv
import mlflow
import os
from data_handler import DataHandler
from mlflow_utils import log_gpu_stats, log_model_histograms, start_mlflow


def should_train_discriminator(disc_loss: float, writer, epoch, disc_loss_target: float, sharpness: float) -> bool:
    """
    Calcula una probabilidad suave de entrenar el discriminador.

    - Si sharpness es alto (e.g. 10), el cambio es brusco (como una puerta).
    - Si sharpness es bajo (e.g. 2), el cambio es progresivo (margen amplio).

    Devuelve un número entre 0 y 1.
    """

    # Normalizamos respecto a los objetivos
    disc_factor = max(0.0, 1.0 - (disc_loss / disc_loss_target))

    # Combinamos (puedes ajustar pesos si quieres)
    score = disc_factor

    # Aplicamos función sigmoide controlada por sharpness
    probability = 1 / (1 + math.exp(-sharpness * (score - 0.5)))

    # Asegurar que está en [0, 1]
    probability = min(max(probability, 0.0), 1.0)

    writer.add_scalar("Debug/Discriminator_Train_Prob", probability, epoch)

    return torch.rand(1).item() < probability


def evaluate_step(encoder, discriminator, test_loader, writer, device, epoch):
    encoder.eval()
    discriminator.eval()

    total_adv_loss = 0
    total_disc_loss = 0
    num_batches = 0

    with torch.no_grad(), amp.autocast("cuda"):
        for i, (images, messages) in enumerate(test_loader):
            images = images.to(device)
            messages = messages.to(device)

            # Forward
            stego_images = encoder(images, messages)
            disc_pred = discriminator(stego_images)

            # Métricas
            disc_real = discriminator(images)
            disc_fake = discriminator(stego_images)

            disc_loss = F.mse_loss(disc_real, torch.ones_like(disc_real)) + \
                        F.mse_loss(disc_fake, torch.zeros_like(disc_fake))

            adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))

            total_adv_loss += adv_loss.detach().item()
            total_disc_loss += disc_loss.item()


            num_batches += 1

        avg_adv_loss = total_adv_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches

        # MLFlow logging por epoch
        mlflow.log_metric("Test/Loss/Discriminator", avg_disc_loss, step=epoch)
        mlflow.log_metric("Test/Loss/Adversarial", avg_adv_loss, step=epoch)

        # TensorBoard logging por epoch
        writer.add_scalar("Test/Loss/Discriminator", avg_disc_loss, epoch)
        writer.add_scalar("Test/Loss/Adversarial", avg_adv_loss, epoch)


        # Imágenes
        images_01 = (images + 1) / 2
        stego_images_01 = (stego_images + 1) / 2
        img_grid_real = make_grid(images_01[:8].cpu(), nrow=4, normalize=True)
        img_grid_stego = make_grid(stego_images_01[:8].cpu(), nrow=4, normalize=True)
        writer.add_image("Test/Images/Real", img_grid_real, epoch)
        writer.add_image("Test/Images/Stego", img_grid_stego, epoch)

        # Debug de diferencias entre stego-images
        img = images[:2]  # coge dos imágenes del batch
        stego = stego_images[:2]

        diff = (stego[0] - stego[1]).abs().mean().item()
        diff_map = ((stego[:1] - img[:1]) ** 2).mean(dim=1, keepdim=True)
        norm_diff = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

        writer.add_image("Test/Real", img[0].cpu(), epoch)
        writer.add_image("Test/Stego", stego[0].cpu(), epoch)
        writer.add_image("Test/Stego_vs_Real_DiffMap", diff_map[0], epoch)
        writer.add_image("Test/NormalizedDiffMap", norm_diff[0], epoch)
        writer.add_scalar("Test/Stego1_vs_Stego2_MsgDiff", diff, epoch)
        mlflow.log_metric("Test/Stego1_vs_Stego2_MsgDiff", diff, step=epoch)

    encoder.train()
    discriminator.train()

def get_noisy(image, noise_std):
    if noise_std > 0:
        noise = torch.randn_like(image) * noise_std
        _image = image + noise
        _image = torch.clamp(_image, -1, 1)

        return _image
    else:
        return image

def train_discriminator_step(discriminator, disc_opt, scaler, scheduler_disc, images, stego_images):
    disc_real = discriminator(images)
    disc_fake = discriminator(stego_images.detach())

    disc_loss = BCEWithLogitsLoss(disc_real, torch.ones_like(disc_real)) + \
                BCEWithLogitsLoss(disc_fake, torch.zeros_like(disc_fake))

    disc_opt.zero_grad()
    scaler.scale(disc_loss).backward()
    scaler.step(disc_opt)
    scaler.update()

    # Solo avanzar el scheduler si hubo grads válidos
    if any(p.grad is not None for p in discriminator.parameters()):
        scheduler_disc.step()

    return disc_loss

def train_step(epoch, images, messages, encoder, discriminator, train_discriminator, disc_opt, scheduler_disc, enc_dec_opt,
               scheduler_enc_dec, scaler):

    # Forward
    with amp.autocast("cuda"):
        stego_images = encoder(images, messages)

        if train_discriminator:
            disc_loss = train_discriminator_step(discriminator, disc_opt, scaler, scheduler_disc, images, stego_images)
        else:
            disc_loss = 0  # para no joder la media, el numero de batches tampoco va a subir

        disc_pred = discriminator(stego_images)
        adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))

        enc_dec_opt.zero_grad()
        scaler.scale(adv_loss).backward()
        scaler.step(enc_dec_opt)
        scheduler_enc_dec.step()
        scaler.update()
        log_gpu_stats(mlflow=mlflow, epoch=epoch)

    return adv_loss, disc_loss, stego_images

def log_epoch(writer, epoch, avg_disc_loss, avg_adv_loss, images, stego_images, global_step,
              current_lr_enc_dec, current_lr_disc):
    # MLFlow logging por epoch
    mlflow.log_metric("Loss/Discriminator", avg_disc_loss, step=epoch)
    mlflow.log_metric("Loss/Adversarial", avg_adv_loss, step=epoch)

    # TensorBoard logging por epoch
    writer.add_scalar("Loss/Discriminator", avg_disc_loss, epoch)
    writer.add_scalar("Loss/Adversarial", avg_adv_loss, epoch)

    images_01 = (images + 1) / 2
    stego_images_01 = (stego_images + 1) / 2
    img_grid_real = make_grid(images_01[:8].cpu(), nrow=4, normalize=False)
    img_grid_stego = make_grid(stego_images_01[:8].detach().cpu(), nrow=4, normalize=False)

    writer.add_image("Images/Real", img_grid_real, epoch)
    writer.add_image("Images/Stego", img_grid_stego, epoch)

    writer.add_scalar('LR/EncDec', current_lr_enc_dec, global_step)
    writer.add_scalar('LR/Disc', current_lr_disc, global_step)

    mlflow.log_metric('LR/EncDec', current_lr_enc_dec, step=global_step)
    mlflow.log_metric('LR/Disc', current_lr_disc, step=global_step)

def save_models(epoch, encoder, discriminator, scaler, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder._orig_mod.state_dict(),
        "discriminator_state_dict": discriminator._orig_mod.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(checkpoint, path)
    mlflow.log_artifact(path)

    latest_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_latest.pt")
    torch.save(checkpoint, latest_path)
    mlflow.log_artifact(latest_path)
    print(f"✅ Modelos guardados correctamente en {path}")

def start(device, warm_up_len, image_loss_lambda, freeze_disc_loss, image_channels, image_size, batch_size, num_epochs,
          image_input_res, epochs_to_val, epochs_to_save, noise_std, style_loss_weight, disc_loss_target, sharpness,
          run_name, checkpoint_dir, log_dir, message_size, pct_start):

    train_dataset, train_loader, test_dataset, test_loader = DataHandler(batch_size=batch_size).get()

    encoder = Encoder(image_channels=image_channels, message_size=message_size).to(device)
    discriminator = Discriminator(image_channels=image_channels).to(device)

    # === Optimizadores ===
    enc_dec_opt = torch.optim.Adam(list(encoder.parameters()), lr=1e-4)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # === Schedulers ===
    _steps_per_epoch = len(train_loader)
    scheduler_enc_dec = OneCycleLR(enc_dec_opt, max_lr=1e-4, steps_per_epoch=_steps_per_epoch, epochs=num_epochs, pct_start=0.1, anneal_strategy='cos')
    scheduler_disc = OneCycleLR(disc_opt, max_lr=1e-4, steps_per_epoch=_steps_per_epoch, epochs=num_epochs, pct_start=0.1, anneal_strategy='cos')

    # Scaler
    scaler = amp.GradScaler()

    # Compilar pesos
    encoder = torch.compile(encoder)
    discriminator = torch.compile(discriminator)

    # Config MLFlow
    _ = start_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "warm_up_len": warm_up_len,
            "batch_size": batch_size,
            "image_loss_lambda": image_loss_lambda,
            "freeze_disc_loss": freeze_disc_loss,
            "image_channels": image_channels,
            "image_size": image_size,
            "num_epochs": num_epochs,
            "image_input_res": image_input_res,
            "noise_std": noise_std,
            "style_loss_weight": style_loss_weight,
            "EPOCHS_TO_VAL": epochs_to_val,
            "EPOCHS_TO_SAVE": epochs_to_save,
            "disc_loss_target": disc_loss_target,
            "sharpness": sharpness,
            "message_size": message_size,
            "scheduler": "OneCycleLR",
            "pct_start": pct_start,
            "anneal_strategy": "cos"
        },)

        train_model(
            device=device,
            start_epoch=0,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            encoder=encoder,
            discriminator=discriminator,
            scaler=scaler,
            scheduler_enc_dec=scheduler_enc_dec,
            scheduler_disc=scheduler_disc,
            disc_opt=disc_opt,
            disc_loss_target=disc_loss_target,
            sharpness=sharpness,
            enc_dec_opt=enc_dec_opt,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )


def train_model(device, start_epoch, num_epochs, train_loader, test_loader, encoder, discriminator, scaler, scheduler_enc_dec,
                scheduler_disc, disc_opt, enc_dec_opt, checkpoint_dir, log_dir, disc_loss_target, sharpness):
    writer = SummaryWriter(log_dir)
    writer.add_text("Entrenamiento", "Iniciado correctamente", 0)
    writer.flush()

    global_step = 0
    train_discriminator = False

    for epoch in range(start_epoch, num_epochs):
        total_disc_loss = 0
        total_adv_loss = 0
        num_batches = 0
        disc_batches = 0

        for i, (images, messages) in enumerate(train_loader):
            images = images.to(device)

            adv_loss, disc_loss, stego_images = train_step(
                epoch=epoch,
                images=images,
                messages=messages,
                encoder=encoder,
                discriminator=discriminator,
                train_discriminator=train_discriminator,
                disc_opt=disc_opt,
                scheduler_disc=scheduler_disc,
                enc_dec_opt=enc_dec_opt,
                scheduler_enc_dec=scheduler_enc_dec,
                scaler=scaler
            )

            total_adv_loss += adv_loss.item()
            total_disc_loss += disc_loss.item()
            num_batches += 1
            global_step += 1

        # Termina de entrenar este epoch
        if train_discriminator:
            disc_batches += 1

        avg_disc_loss = total_disc_loss / disc_batches

        disc_train_next = should_train_discriminator(
            disc_loss=avg_disc_loss,
            writer=writer,
            epoch=epoch,
            disc_loss_target=disc_loss_target,
            sharpness=sharpness
        )

        if train_discriminator and not disc_train_next:
            print(f"🧠 [Discriminador]: Paro de entrenar. disc_loss: {avg_disc_loss}")
            train_discriminator = False
        else: # si no se ha entrenado este epoch
            if epoch > WARM_UP_LEN and disc_train_next:
                print("🧠 [Discriminador]: empiezo a entrenar")
                train_discriminator = True

        avg_disc_loss = total_disc_loss / disc_batches
        avg_adv_loss = total_adv_loss / num_batches

        if (epoch + 1) % EPOCHS_TO_VAL == 0:
            evaluate_step(encoder, discriminator, test_loader, writer, device, epoch)
            print("Evaluando sobre el test wey")


        current_lr_enc_dec = scheduler_enc_dec.get_last_lr()[0]
        current_lr_disc = scheduler_disc.get_last_lr()[0]

        log_epoch(
            writer=writer,
            epoch=epoch,
            avg_disc_loss=avg_disc_loss,
            avg_adv_loss=avg_adv_loss,
            images=images,
            stego_images=stego_images,
            global_step=global_step,
            current_lr_enc_dec=current_lr_enc_dec,
            current_lr_disc=current_lr_disc
        )

        log_model_histograms(writer, encoder, "Encoder", epoch)
        log_model_histograms(writer, discriminator, "Discriminator", epoch)

        if (epoch + 1) % EPOCHS_TO_SAVE == 0:
            save_models(epoch, encoder, discriminator, scaler, checkpoint_dir)

        torch.cuda.empty_cache()

    mlflow.end_run()
    writer.close()


if __name__ == "__main__":
    load_dotenv()

    # Cuda
    torch.set_float32_matmul_precision('high')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Parametros de las redes
    WARM_UP_LEN = int(os.getenv("WARM_UP_LEN"))  # numero de epochs que dejo al discriminador sin entrenar
    IMAGE_LOSS_LAMBDA = float(os.getenv("image_loss_lambda"))  # Parametro para darle algo de tolerancia al image loss
    FREEZE_DISC_LOSS = float(
        os.getenv("FREEZE_DISC_LOSS"))  # loss maximo que alcanza el discriminador antes de ser congelado
    IMAGE_CHANNELS = int(os.getenv("image_channels"))
    IMAGE_SIZE = int(os.getenv("image_size"))
    BATCH_SIZE = int(os.getenv("batch_size"))
    NUM_EPOCHS = int(os.getenv("num_epochs"))
    IMAGE_INPUT_RES = int(os.getenv("IMAGE_INPUT_RES"))  # resolucion de la imagen de entrada
    EPOCHS_TO_VAL = int(os.getenv("EPOCHS_TO_VAL"))  # numero de epochs entre validaciones
    EPOCHS_TO_SAVE = int(os.getenv("EPOCHS_TO_SAVE"))  # numero de epochs para guardar el modelo
    NOISE_STD = float(os.getenv(
        "noise_std"))  # ruido que se le mete a la imagen generada. Entre 0.01 y 0.05 es razonable para imágenes normalizadas
    STYLE_LOSS_WEIGHT = float(os.getenv("style_loss_weight"))
    DISC_LOSS_TARGET = float(os.getenv("disc_loss_target"))
    SHARPNESS = float(os.getenv("sharpness"))
    MESSAGE_SIZE = int(os.getenv("message_size"))
    PCT_START = float(os.getenv("pct_start"))
    RUN_NAME = os.getenv("RUN_NAME")

    CHECKPOINT_DIR = os.path.join(os.getenv("checkpoint_dir"), RUN_NAME)
    LOG_DIR = os.path.join(os.getenv("log_dir"), RUN_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start(
        device=DEVICE,
        warm_up_len=WARM_UP_LEN,
        image_loss_lambda=IMAGE_LOSS_LAMBDA,
        freeze_disc_loss=FREEZE_DISC_LOSS,
        image_channels=IMAGE_CHANNELS,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        image_input_res=IMAGE_INPUT_RES,
        epochs_to_val=EPOCHS_TO_VAL,
        epochs_to_save=EPOCHS_TO_SAVE,
        noise_std=NOISE_STD,
        style_loss_weight=STYLE_LOSS_WEIGHT,
        disc_loss_target=DISC_LOSS_TARGET,
        sharpness=SHARPNESS,
        run_name=RUN_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        message_size=MESSAGE_SIZE,
        pct_start=PCT_START
    )

