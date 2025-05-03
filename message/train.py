import os
import re
import random
import torch
import torch.nn as nn
import evaluate
from torch import amp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformer_decoder import TransformerDecoder
from dotenv import load_dotenv
import mlflow

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def start_mlflow():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("DATABRICKS_HOST"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DATABRICKS_USER_NAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DATABRICKS_TOKEN")

    databricks_user = os.getenv("DATABRICKS_USER_NAME")
    experiment_name = os.getenv("EXPERIMENT_NAME")

    # Ruta completa del experimento
    experiment_path = f"/Users/{databricks_user}/{experiment_name}"

    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    print("Usando experimento:", experiment_path)

    mlflow.set_experiment(experiment_path)

    return mlflow

def train_step(batch, bert, decoder, criterion, optimizer, pad_token_id, sos_token_id, vocab_size, epoch, writer):
    input_ids = batch["input_ids"].to(DEVICE)
    targets = input_ids[:, 1:]

    attention_mask = (input_ids != pad_token_id).long()

    with torch.no_grad(), amp.autocast("cuda"):
        z = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)

    outputs = decoder(z, sos_token_id=sos_token_id, targets=targets, generate=False, teacher_forcing_ratio=0.7)
    targets = targets[:, :outputs.size(1)]

    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()

    if (epoch + 1) % 10 == 0:
        for name, param in decoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                writer.add_histogram(f"Decoder/Weights/{name}", param.data, epoch)
                writer.add_histogram(f"Decoder/Grads/{name}", param.grad, epoch)

    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def validate_step(writer, epoch, bert, decoder, tokenizer, pad_token_id, sos_token_id, val_dataset):
    decoder.eval()
    bert.eval()

    rouge_metric = evaluate.load("rouge")
    smoother = SmoothingFunction().method1

    sample_size = min(10, len(val_dataset))
    sample_batch = val_dataset.select(random.sample(range(len(val_dataset)), sample_size))

    refs, hyps, bleus, rouges = [], [], [], []

    for example in sample_batch:
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        val_input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(DEVICE)

        attention_mask = (val_input_ids != pad_token_id).long()

        with torch.no_grad():
            z = bert(input_ids=val_input_ids, attention_mask=attention_mask).last_hidden_state
            output_ids = decoder(z, generate=True, sos_token_id=sos_token_id, eos_token_id=tokenizer.eos_token_id)
            decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        ref = [input_text.split()]
        hyp = decoded_text.split()
        bleu = sentence_bleu(ref, hyp, smoothing_function=smoother)
        rouge_score = rouge_metric.compute(predictions=[decoded_text], references=[input_text])

        refs.append(input_text)
        hyps.append(decoded_text)
        bleus.append(bleu)
        rouges.append(rouge_score)

    # Evaluación fija con frase conocida
    fixed_sentence = "the quick brown fox jumps over the lazy dog"
    fixed_ids = tokenizer.encode(fixed_sentence, truncation=True, max_length=tokenizer.model_max_length, padding="max_length")
    fixed_tensor = torch.tensor(fixed_ids).unsqueeze(0).to(DEVICE)
    fixed_mask = (fixed_tensor != pad_token_id).long()

    with torch.no_grad():
        z_fixed = bert(input_ids=fixed_tensor, attention_mask=fixed_mask).last_hidden_state
        output_ids = decoder(z_fixed, generate=True, sos_token_id=sos_token_id, eos_token_id=tokenizer.eos_token_id)
        decoded_fixed = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    writer.add_text("Fixed/Reference", fixed_sentence, epoch)
    writer.add_text("Fixed/Hypothesis", decoded_fixed, epoch)

    avg_bleu = sum(bleus) / sample_size
    avg_rouge1 = sum(r["rouge1"] for r in rouges) / sample_size
    avg_rougel = sum(r["rougeL"] for r in rouges) / sample_size

    writer.add_text("Sample/Reference", refs[0], epoch)
    writer.add_text("Sample/Hypothesis", hyps[0], epoch)

    return avg_bleu, avg_rouge1, avg_rougel

def get_last_checkpoint(checkpoint_dir, decoder, optimizer, scheduler):
    # Buscar todos los archivos .pt con un patrón de número de epoch
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and re.search(r'\d+', f)
    ]

    if not checkpoint_files:
        print("ℹ️ No se encontró ningún checkpoint. Comenzando desde cero.")
        return decoder, optimizer, scheduler, 0

    # Ordenar por número extraído del nombre del archivo
    checkpoint_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    latest_ckpt = checkpoint_files[-1]
    path = os.path.join(checkpoint_dir, latest_ckpt)

    try:
        checkpoint = torch.load(path)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✅ Reanudado desde {path}")
        return decoder, optimizer, scheduler, start_epoch
    except Exception as e:
        print(f"❌ Error al cargar el checkpoint {path}: {e}")
        return decoder, optimizer, scheduler, 0

def save_checkpoint(checkpoint_dir, epoch, decoder, optimizer, scheduler):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"decoder_epoch{epoch}.pt")

    torch.save({
        'epoch': epoch,
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)

    print(f"💾 Checkpoint guardado en {checkpoint_path}")

def train_model(device, tokenizer, train_loader, val_dataset, decoder, optimizer, scheduler, total_epochs, vocab_size,
                validate_every, save_every, checkpoint_dir, log_dir):
    writer = SummaryWriter(log_dir)
    writer.add_text("Entrenamiento", "Iniciado correctamente", 0)
    writer.flush()

    sos_token_id = tokenizer.cls_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # o '[PAD]'

    bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    pad_token_id = tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)

    decoder, optimizer, scheduler, start_epoch = get_last_checkpoint(checkpoint_dir, decoder, optimizer, scheduler)

    for epoch in range(start_epoch, total_epochs + 1):
        decoder.train()

        total_loss = 0
        for batch in train_loader:
            loss = train_step(
                batch=batch,
                bert=bert,
                decoder=decoder,
                criterion=criterion,
                optimizer=optimizer,
                pad_token_id=pad_token_id,
                sos_token_id=sos_token_id,
                vocab_size=vocab_size,
                epoch=epoch,
                writer=writer
            )
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        ppl = torch.exp(torch.tensor(avg_loss))

        print(f"📚 Epoch {epoch}/{total_epochs} — Loss: {avg_loss:.4f} — Perplexity: {ppl:.2f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Perplexity/train", ppl, epoch)
        total_norm = sum(p.grad.detach().norm().item() for p in decoder.parameters() if p.grad is not None)
        writer.add_scalar("GradientNorm", total_norm, epoch)

        scheduler.step()
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        mlflow.log_metrics({
            "Loss/train": avg_loss,
            "Perplexity/train": ppl,
            "LR": scheduler.get_last_lr()[0],
            "Perplexity": ppl.item(),
            "GradientNorm": total_norm
        }, step=epoch)

        # Validación
        if epoch % validate_every == 0:
            avg_bleu, avg_rouge1, avg_rougel = validate_step(
                writer=writer,
                epoch=epoch,
                bert=bert,
                decoder=decoder,
                tokenizer=tokenizer,
                pad_token_id=pad_token_id,
                sos_token_id=sos_token_id,
                val_dataset=val_dataset
            )
            writer.add_scalar("BLEU/val", avg_bleu, epoch)
            writer.add_scalar("ROUGE1/val", avg_rouge1, epoch)
            writer.add_scalar("ROUGE-L/val", avg_rougel, epoch)

            mlflow.log_metrics({
                "BLEU": avg_bleu,
                "ROUGE1": avg_rouge1,
                "ROUGEL": avg_rougel,
            }, step=epoch)

        # Checkpoint
        if epoch % save_every == 0 or epoch == total_epochs:
            save_checkpoint(checkpoint_dir, epoch, decoder, optimizer, scheduler)

    writer.close()

def start(device, batch_size, run_name, checkpoint_dir, log_dir, embed_dim, epochs,
          validate_every, save_every, max_token_len, max_sencence_sample_len):

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = tokenizer.vocab_size

    decoder = TransformerDecoder(embedding_dim=embed_dim, vocab_size=vocab_size, max_len=max_token_len).to(
        device)

    decoder = torch.compile(decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    # --- Dataset ---
    print("🔄 Cargando dataset...")
    raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")['train']
    dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, val_dataset = dataset["train"], dataset["test"]

    train_dataset = train_dataset.filter(lambda x: x["text"].strip() != "" and 10 < len(x["text"].split()) < max_sencence_sample_len)
    val_dataset = val_dataset.filter(lambda x: x["text"].strip() != "" and 10 < len(x["text"].split()) < max_sencence_sample_len)

    train_dataset = train_dataset.map(
        lambda x: {"input_ids": tokenizer.encode(x["text"], truncation=True, max_length=max_token_len, padding="max_length")})
    val_dataset = val_dataset.map(
        lambda x: {"input_ids": tokenizer.encode(x["text"], truncation=True, max_length=max_token_len, padding="max_length")})

    train_dataset.set_format(type="torch", columns=["input_ids"])
    val_dataset.set_format(type="torch", columns=["input_ids"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    _ = start_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "max_token_len": max_token_len,
            "max_sencence_sample_len": max_sencence_sample_len,
            "epochs": epochs,
            "validate_every": validate_every,
            "save_every": save_every
        }, )

        train_model(
            device=device,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_dataset=val_dataset,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            total_epochs=epochs,
            vocab_size=vocab_size,
            validate_every=validate_every,
            save_every=save_every,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )


if __name__ == "__main__":
    load_dotenv()

    torch.set_float32_matmul_precision('high')
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = int(os.getenv("batch_size"))
    EMBED_DIM = int(os.getenv("message_size"))
    MAX_TOKEN_LEN = int(os.getenv("MAX_TOKEN_LEN"))
    MAX_SENCTENCE_SAMPLE_LEN = int(os.getenv("MAX_SENCTENCE_SAMPLE_LEN"))
    EPOCHS = int(os.getenv("num_epochs"))

    VALIDATE_EVERY = int(os.getenv("EPOCHS_TO_VAL"))  # numero de epochs entre validaciones
    SAVE_EVERY = int(os.getenv("EPOCHS_TO_SAVE"))  # numero de epochs para guardar el modelo
    RUN_NAME = os.getenv("RUN_NAME")

    CHECKPOINT_DIR = os.path.join(os.getenv("checkpoint_dir"), RUN_NAME)
    LOG_DIR = os.path.join(os.getenv("log_dir"), RUN_NAME)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    start(
        device=DEVICE,
        batch_size=BATCH_SIZE,
        run_name=RUN_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        embed_dim=EMBED_DIM,
        max_token_len=MAX_TOKEN_LEN,
        max_sencence_sample_len=MAX_SENCTENCE_SAMPLE_LEN,
        epochs=EPOCHS,
        validate_every=VALIDATE_EVERY,
        save_every=SAVE_EVERY
    )
