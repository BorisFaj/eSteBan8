import torch
from transformers import AutoTokenizer
from text_coder import TextCompressorVAE
from text_decoder import TextDecoder

# --- Configuración ---
T = 512                # Tamaño del vector comprimido
MAX_LEN = 60           # Longitud máxima del texto generado
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Carga tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
sos_token_id = tokenizer.cls_token_id

# --- Reconstruye modelos y carga pesos ---
compressor = TextCompressorVAE(latent_dim=T).to(DEVICE)
decoder = TextDecoder(
    embedding_dim=256,
    hidden_dim=T,
    vocab_size=vocab_size,
    max_len=MAX_LEN
).to(DEVICE)

compressor.load_state_dict(torch.load("models/message/compressor_epoch1600.pt", map_location=DEVICE), strict=False)
decoder.load_state_dict(torch.load("models/message/decoder_epoch1600.pt", map_location=DEVICE), strict=False)

compressor.eval()
decoder.eval()

# --- Texto de entrada ---
input_text = "= = = 2010 : fourth australian open = = ="
print(f"🔒 Texto original: {input_text}")

# --- Codifica directamente (sin tokenizar manualmente) ---
with torch.no_grad():
    compressed_vector, *_ = compressor([input_text])

compressed_vector = compressed_vector.to(DEVICE)

# --- Decodifica ---
with torch.no_grad():
    logits = decoder(compressed_vector, sos_token_id=sos_token_id)  # (1, 60, vocab_size)
    token_ids = torch.argmax(logits, dim=-1)  # (1, 60)

# --- Reconstruye texto ---
decoded_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
print(f"🔓 Texto reconstruido: {decoded_text}")
