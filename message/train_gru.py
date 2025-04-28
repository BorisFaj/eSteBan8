import os
import random
import torch
import torch.nn as nn
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from text_coder import TextCompressorVAE
from lstm_attention import LSTMAttention


smoother = SmoothingFunction().method1

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LATENT_DIM = 768
EMBED_DIM = 768
HIDDEN_DIM = 768
MAX_LEN = 60
EPOCHS = 3000
RUN_NAME = "eSteBert_v1.3s"

log_dir = f'./runs/{RUN_NAME}'
checkpoint_dir = f'./checkpoints/{RUN_NAME}'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# --- TensorBoard ---
writer = SummaryWriter(log_dir=log_dir)
rouge = evaluate.load("rouge")

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
sos_token_id = tokenizer.cls_token_id
pad_token_id = tokenizer.pad_token_id

# --- Modelos ---
compressor = TextCompressorVAE(latent_dim=LATENT_DIM, pooling='cls', freeze_bert=True).to(DEVICE)
decoder = LSTMAttention(embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab_size=vocab_size, max_len=MAX_LEN).to(DEVICE)

# --- Dataset ---
print("🔄 Cargando dataset...")
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")['train']
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, val_dataset = dataset["train"], dataset["test"]

train_dataset = train_dataset.filter(lambda x: x["text"].strip() != "" and 10 < len(x["text"].split()) < 50)
val_dataset = val_dataset.filter(lambda x: x["text"].strip() != "" and 10 < len(x["text"].split()) < 50)

train_dataset = train_dataset.map(lambda x: {"input_ids": tokenizer.encode(x["text"], truncation=True, max_length=MAX_LEN, padding="max_length")})
val_dataset = val_dataset.map(lambda x: {"input_ids": tokenizer.encode(x["text"], truncation=True, max_length=MAX_LEN, padding="max_length")})

train_dataset.set_format(type="torch", columns=["input_ids"])
val_dataset.set_format(type="torch", columns=["input_ids"])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Optimización ---
optimizer = torch.optim.Adam(list(decoder.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

# --- Entrenamiento ---
for epoch in range(EPOCHS):
    total_loss = 0
    compressor.eval()
    decoder.train()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        targets = input_ids[:, 1:].contiguous()

        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        z, mu, logvar = compressor(texts)

        outputs = decoder(z, sos_token_id=sos_token_id, targets=targets, generate=False, teacher_forcing_ratio=0.7)
        outputs = outputs[:, :-1, :]
        targets = targets[:, :outputs.size(1)]  # Asegurar igual longitud

        ce_loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = ce_loss + 0.01 * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        ppl = torch.exp(loss)

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"📚 Epoch {epoch + 1}/{EPOCHS} — Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/CE_train", ce_loss.item(), epoch)
    writer.add_scalar("Loss/KL_train", kl_loss.item(), epoch)
    writer.add_scalar("Perplexity/train", ppl.item(), epoch)

    # --- Validación ---
    compressor.eval()
    decoder.eval()
    sample_size = 10
    sample_batch = val_dataset.select(random.sample(range(len(val_dataset)), sample_size))

    refs, hyps, bleus, rouges = [], [], [], []

    for example in sample_batch:
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        val_input_ids = example["input_ids"].clone().detach().unsqueeze(0).to(DEVICE)

        val_targets = val_input_ids[:, 1:]  # igual que en training
        with torch.no_grad():
            z, _, _ = compressor(input_text)
            z = z.to(DEVICE)
            output_logits = decoder(z, generate=True, sos_token_id=sos_token_id)
            output_ids = torch.argmax(output_logits, dim=-1)
            decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            val_logits = decoder(z, sos_token_id=sos_token_id, targets=val_targets, generate=False)[:, :-1, :]
            val_targets = val_targets[:, :val_logits.size(1)]

            val_loss = criterion(val_logits.reshape(-1, vocab_size), val_targets.reshape(-1))
            val_ppl = torch.exp(val_loss)

        ref = [input_text.split()]
        hyp = decoded_text.split()
        bleu = sentence_bleu(ref, hyp, smoothing_function=smoother)
        rouge_score = rouge.compute(predictions=[decoded_text], references=[input_text])

        refs.append(input_text)
        hyps.append(decoded_text)
        bleus.append(bleu)
        rouges.append(rouge_score)

    writer.add_scalar("BLEU/val_avg", sum(bleus) / sample_size, epoch)
    writer.add_scalar("ROUGE1/val_avg", sum(r["rouge1"] for r in rouges) / sample_size, epoch)
    writer.add_scalar("ROUGE-L/val_avg", sum(r["rougeL"] for r in rouges) / sample_size, epoch)
    writer.add_scalar("Perplexity/val", val_ppl.item(), epoch)

    for i, bleu in enumerate(bleus):
        if bleu < 0.3:
            writer.add_text(f"🔴 Reconstruction (BLEU={bleu:.2f})", f"Input: {refs[i]}\nOutput: {hyps[i]}", epoch)

    if epoch % 2 == 0:
        with torch.no_grad():
            sample_batch = val_dataset.select(range(min(100, len(val_dataset))))
            sample_texts = [tokenizer.decode(x["input_ids"], skip_special_tokens=True) for x in sample_batch]
            z_batch, _, _ = compressor(sample_texts)
            z_batch = z_batch.cpu()

            writer.add_embedding(z_batch, metadata=sample_texts, tag="Embeddings", global_step=epoch)

    for name, param in decoder.named_parameters():
        writer.add_histogram(f"Decoder/{name}", param, epoch)

    if (epoch + 1) % 400 == 0 or (epoch + 1) == EPOCHS:
        torch.save(compressor.state_dict(), os.path.join(checkpoint_dir, f"compressor_epoch{epoch+1}.pt"))
        torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, f"decoder_epoch{epoch+1}.pt"))

# --- Guardado final ---
torch.save(compressor.state_dict(), os.path.join(checkpoint_dir, f"compressor_epoch{epoch + 1}.pt"))
torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, f"decoder_epoch{epoch + 1}.pt"))
writer.close()
