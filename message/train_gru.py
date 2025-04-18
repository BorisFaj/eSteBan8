import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from torch.utils.tensorboard import SummaryWriter
import random

from text_coder import TextCompressor
from text_decoder import TextDecoder
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import os

os.makedirs("checkpoints", exist_ok=True)

rouge = evaluate.load("rouge")

# --- TensorBoard ---
writer = SummaryWriter(log_dir="../runs/text_autoencoder")

# --- Configuración ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
EMBED_DIM = 128
LATENT_DIM = 512  # Longitud vector comprimido z
HIDDEN_DIM = LATENT_DIM  # Capa de apoyo a la capa latente
MAX_LEN = 60  # LEN maxima de las frases para entrenar y generar.
EPOCHS = 300

# --- Tokenizer y vocabulario ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
sos_token_id = tokenizer.cls_token_id
pad_token_id = tokenizer.pad_token_id

# --- Modelos ---
compressor = TextCompressor(output_dim=LATENT_DIM, pooling='cls', freeze_bert=True).to(DEVICE)
decoder = TextDecoder(embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab_size=vocab_size, max_len=MAX_LEN).to(DEVICE)

# --- Datasets ---
print("🔄 Cargando dataset...")

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]

# Divide 90% train, 10% val
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

def clean(example):
    text = example["text"].strip()
    return text != "" and 10 < len(text.split()) < 50

train_dataset = train_dataset.filter(clean)
val_dataset = val_dataset.filter(clean)


def preprocess(example):
    tokens = tokenizer.encode(example["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
    return {"input_ids": tokens}

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

train_dataset.set_format(type="torch", columns=["input_ids"])
val_dataset.set_format(type="torch", columns=["input_ids"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)  # no hace falta batch grande para validar

# --- Optimización ---
optimizer = torch.optim.Adam(list(decoder.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# --- Entrenamiento ---
for epoch in range(EPOCHS):
    total_loss = 0
    compressor.eval()
    decoder.train()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)

        with torch.no_grad():
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            vecs = compressor(texts)  # (B, latent_dim)

        # Quita el primer token (sos) de target, ya que no se debe predecir
        targets = input_ids[:, 1:].contiguous()
        # Quita el último token del output para alinear
        outputs = decoder(vecs, sos_token_id=sos_token_id, generate=False)[:, :-1, :].contiguous()

        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"📚 Epoch {epoch + 1}/{EPOCHS} — Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # --- Guardado de checkpoint cada 400 epochs ---
    if (epoch + 1) % 400 == 0 or (epoch + 1) == EPOCHS:
        torch.save(compressor.state_dict(), f"checkpoints/compressor_epoch{epoch + 1}.pt")
        torch.save(decoder.state_dict(), f"checkpoints/decoder_epoch{epoch + 1}.pt")
        print(f"💾 Checkpoint guardado en epoch {epoch + 1}")

    # --- Log texto original y reconstruido (VALIDACIÓN) ---
    compressor.eval()
    decoder.eval()

    val_sample = random.choice(val_dataset)
    input_text = tokenizer.decode(val_sample["input_ids"], skip_special_tokens=True)
    with torch.no_grad():
        z = compressor(input_text).unsqueeze(0).to(DEVICE)
        output_logits = decoder(z, generate=True)  # (1, max_len, vocab)
        output_ids = torch.argmax(output_logits, dim=-1)
        decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # BLEU
    ref = [input_text.split()]
    hyp = decoded_text.split()
    bleu_score = sentence_bleu(ref, hyp)
    writer.add_scalar("BLEU/val", bleu_score, epoch)

    # ROUGE
    rouge_score = rouge.compute(predictions=[decoded_text], references=[input_text])
    writer.add_scalar("ROUGE1/val", rouge_score["rouge1"], epoch)
    writer.add_scalar("ROUGE-L/val", rouge_score["rougeL"], epoch)

    writer.add_text("Original", input_text, epoch)
    writer.add_text("Reconstruido", decoded_text, epoch)

    # --- Embeddings latentes (cada 2 epochs) ---
    if epoch % 2 == 0:
        with torch.no_grad():
            sample_batch = val_dataset.select(range(min(100, len(val_dataset))))
            sample_texts = [tokenizer.decode(x["input_ids"], skip_special_tokens=True) for x in sample_batch]
            z_batch = compressor(sample_texts).cpu()
            writer.add_embedding(z_batch, metadata=sample_texts, tag="Embeddings", global_step=epoch)

    # --- Histograma de pesos ---
    for name, param in decoder.named_parameters():
        writer.add_histogram(f"Decoder/{name}", param, epoch)

# --- Guardado de modelos ---
torch.save(compressor.state_dict(), "../compressor.pt")
torch.save(decoder.state_dict(), "../decoder.pt")
writer.close()

