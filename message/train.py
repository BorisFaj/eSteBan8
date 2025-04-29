import os
import random
import torch
import torch.nn as nn
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from lstm_attention import LSTMAttention

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
EMBED_DIM = 768
HIDDEN_DIM = 768
MAX_LEN = 60
EPOCHS = 3000
VALIDATE_EVERY = 10
SAVE_EVERY = 400
RUN_NAME = "eSteBert_v1.4"
LOG_DIR = f'./runs/{RUN_NAME}'
CKPT_DIR = f'./checkpoints/{RUN_NAME}'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# --- Setup ---
writer = SummaryWriter(log_dir=LOG_DIR)
rouge_metric = evaluate.load("rouge")
smoother = SmoothingFunction().method1

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
sos_token_id = tokenizer.cls_token_id
pad_token_id = tokenizer.pad_token_id

bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
decoder = LSTMAttention(embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab_size=vocab_size, max_len=MAX_LEN).to(DEVICE)

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

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

# --- Funciones ---
def train_step(batch):
    input_ids = batch["input_ids"].to(DEVICE)
    targets = input_ids[:, 1:]

    attention_mask = (input_ids != pad_token_id).long()

    with torch.no_grad():
        z = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [CLS]

    outputs = decoder(z, sos_token_id=sos_token_id, targets=targets, generate=False, teacher_forcing_ratio=0.7)
    targets = targets[:, :outputs.size(1)]

    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def validate_step():
    decoder.eval()
    bert.eval()

    sample_size = 10
    sample_batch = val_dataset.select(random.sample(range(len(val_dataset)), sample_size))

    refs, hyps, bleus, rouges = [], [], [], []

    for example in sample_batch:
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        val_input_ids = example["input_ids"].unsqueeze(0).to(DEVICE)

        attention_mask = (val_input_ids != pad_token_id).long()

        with torch.no_grad():
            z = bert(input_ids=val_input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            output_logits = decoder(z, generate=True, sos_token_id=sos_token_id)
            output_ids = torch.argmax(output_logits, dim=-1)
            decoded_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        ref = [input_text.split()]
        hyp = decoded_text.split()
        bleu = sentence_bleu(ref, hyp, smoothing_function=smoother)
        rouge_score = rouge_metric.compute(predictions=[decoded_text], references=[input_text])

        refs.append(input_text)
        hyps.append(decoded_text)
        bleus.append(bleu)
        rouges.append(rouge_score)

    avg_bleu = sum(bleus) / sample_size
    avg_rouge1 = sum(r["rouge1"] for r in rouges) / sample_size
    avg_rougel = sum(r["rougeL"] for r in rouges) / sample_size

    return avg_bleu, avg_rouge1, avg_rougel

# --- Entrenamiento principal ---
for epoch in range(1, EPOCHS + 1):
    decoder.train()

    total_loss = 0
    for batch in train_loader:
        loss = train_step(batch)
        total_loss += loss

    avg_loss = total_loss / len(train_loader)
    ppl = torch.exp(torch.tensor(avg_loss))

    print(f"📚 Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f} — Perplexity: {ppl:.2f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Perplexity/train", ppl, epoch)

    scheduler.step()

    # Validación
    if epoch % VALIDATE_EVERY == 0:
        avg_bleu, avg_rouge1, avg_rougel = validate_step()
        writer.add_scalar("BLEU/val", avg_bleu, epoch)
        writer.add_scalar("ROUGE1/val", avg_rouge1, epoch)
        writer.add_scalar("ROUGE-L/val", avg_rougel, epoch)

    # Checkpoint
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        torch.save(decoder.state_dict(), os.path.join(CKPT_DIR, f"decoder_epoch{epoch}.pt"))

writer.close()
