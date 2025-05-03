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

def sobel_loss(stego, original):
    grad_stego = gradient_magnitude(stego)
    grad_orig = gradient_magnitude(original)
    return F.l1_loss(grad_stego, grad_orig)

def calc_disc_loss(discriminator, images, stego_images):
    pred_real = discriminator(images)
    pred_fake = discriminator(stego_images.detach())

    loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

    disc_loss = loss_real / loss_fake

    print(f"[Disc Real] mean={pred_real.mean().item():.2f} std={pred_real.std().item():.2f}")
    print(f"[Disc Fake] mean={pred_fake.mean().item():.2f} std={pred_fake.std().item():.2f}")

    return disc_loss

def discriminator_step(discriminator, disc_opt, scaler, scheduler_disc, images, stego_images, train):

    disc_loss = calc_disc_loss(discriminator=discriminator, images=images, stego_images=stego_images)

    if train:
        disc_opt.zero_grad()
        scaler.scale(disc_loss).backward()
        scaler.step(disc_opt)

        if any(p.grad is not None for p in discriminator.parameters()):
            scheduler_disc.step()

        scaler.update()

    return disc_loss, scaler, scheduler_disc, disc_opt

def train_step(epoch, images, messages, encoder, discriminator, train_discriminator, disc_opt, scheduler_disc, enc_dec_opt,
               scheduler_enc_dec, scaler, adv_weight, img_weight, edge_weight, disc_weight):
    with amp.autocast("cuda"):
        stego_images = encoder(images, messages)

        disc_loss, scaler, scheduler_disc, disc_opt = discriminator_step(
            discriminator=discriminator,
            disc_opt=disc_opt,
            scaler=scaler,
            scheduler_disc=scheduler_disc,
            images=images,
            stego_images=stego_images,
            train=train_discriminator
        )

        disc_pred = discriminator(stego_images)
        adv_loss = F.mse_loss(disc_pred, torch.ones_like(disc_pred))
        image_loss = F.mse_loss(stego_images, images)
        edge_loss = sobel_loss(stego_images, images)

        total_loss = adv_weight * adv_loss + img_weight * image_loss + edge_weight * edge_loss

        enc_dec_opt.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(enc_dec_opt)
        scheduler_enc_dec.step()
        scaler.update()
        log_gpu_stats(mlflow=mlflow, epoch=epoch)

    return adv_loss, disc_loss, stego_images, scaler, scheduler_disc, scheduler_enc_dec, disc_opt, enc_dec_opt

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

def start(device, warm_up_len, image_loss_lambda, freeze_disc_loss, image_channels, image_size, batch_size, num_epochs,
          image_input_res, epochs_to_val, epochs_to_save, noise_std, style_loss_weight, disc_loss_target, sharpness,
          run_name, checkpoint_dir, log_dir, message_size, pct_start, adv_weight, img_weight, edge_weight, disc_weight):

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
            "anneal_strategy": "cos",
            "adv_weight": adv_weight,
            "img_weight": img_weight,
            "edge_weight": edge_weight,
            "disc_weight": disc_weight
        },)

        train_model(
            device=device,
            start_epoch=0,
            warm_up_len=warm_up_len,
            epochs_to_save=epochs_to_save,
            epochs_to_val=epochs_to_val,
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
            log_dir=log_dir,
            adv_weight=adv_weight,
            img_weight=img_weight,
            edge_weight=edge_weight,
            disc_weight=disc_weight
        )

def to_float(val, default=0.0) -> float:
    if val is None:
        return default
    if isinstance(val, torch.Tensor):
        return float(val.item())
    return float(val)

def train_model(device, start_epoch, num_epochs, train_loader, test_loader, encoder, discriminator, scaler, scheduler_enc_dec,
                scheduler_disc, disc_opt, enc_dec_opt, checkpoint_dir, log_dir, disc_loss_target, sharpness,
                warm_up_len, epochs_to_save, epochs_to_val, adv_weight, img_weight, edge_weight, disc_weight):
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

            (adv_loss, disc_loss, stego_images, scaler, scheduler_disc,
             scheduler_enc_dec, disc_opt, enc_dec_opt) = train_step(
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
                scaler=scaler,
                adv_weight=adv_weight,
                img_weight=img_weight,
                edge_weight=edge_weight,
                disc_weight=disc_weight
            )

            total_adv_loss += to_float(adv_loss)
            total_disc_loss += to_float(disc_loss)
            num_batches += 1
            global_step += 1

        # Termina de entrenar este epoch
        if train_discriminator:
            disc_batches += 1

        # Evalua si toca
        if (epoch + 1) % epochs_to_val == 0:
            evaluate_step(encoder, discriminator, test_loader, writer, device, epoch)
            print("Evaluando sobre el test wey")

        if train_discriminator:
            disc_batches += 1

        avg_disc_loss = total_disc_loss / max(1, disc_batches)
        avg_adv_loss = total_adv_loss / num_batches

        disc_train_next = should_train_discriminator(
            disc_loss=avg_disc_loss,
            gen_loss=avg_adv_loss,
            writer=writer,
            epoch=epoch,
            disc_loss_target=disc_loss_target,
            sharpness=sharpness
        )

        if train_discriminator and not disc_train_next:
            print(f"🧠 [Discriminador]: Paro de entrenar. disc_loss: {avg_disc_loss}")
            train_discriminator = False
        else: # si no se ha entrenado este epoch
            if epoch > warm_up_len and disc_train_next:
                print("🧠 [Discriminador]: empiezo a entrenar")
                train_discriminator = True

        # Log y save
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

        if (epoch + 1) % epochs_to_save == 0:
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
    ADV_WEIGHT = float(os.getenv("adv_weight"))
    IMG_WEIGHT = float(os.getenv("img_weight"))
    EDGE_WEIGHT = float(os.getenv("edge_weight"))
    DISC_WEIGHT = float(os.getenv("disc_weight"))
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
        pct_start=PCT_START,
        adv_weight=ADV_WEIGHT,
        img_weight=IMG_WEIGHT,
        edge_weight=EDGE_WEIGHT,
        disc_weight=DISC_WEIGHT
    )

