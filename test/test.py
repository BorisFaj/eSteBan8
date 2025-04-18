import torch
from message.text_coder import TextCompressor
from message.text_decoder import TextDecoder
from transformers import AutoTokenizer


# --- Configuración ---
T = 256                # Tamaño del vector comprimido
MAX_LEN = 30           # Longitud máxima del texto generado
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Carga tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size
sos_token_id = tokenizer.cls_token_id

# --- Inicializa modelos ---
compressor = TextCompressor(output_dim=T).to(DEVICE).eval()
decoder = TextDecoder(
    embedding_dim=128,
    hidden_dim=T,  # debe coincidir con el tamaño de compresión
    vocab_size=vocab_size,
    max_len=MAX_LEN
).to(DEVICE).eval()

# --- Texto de entrada ---
input_text = "esto es una prueba seria de compresión textual extrema"
print(f"🔒 Texto original: {input_text}")

# --- Codifica ---
with torch.no_grad():
    compressed_vector = compressor(input_text).unsqueeze(0).to(DEVICE)  # (1, T)

# --- Decodifica ---
with torch.no_grad():
    token_ids = decoder(compressed_vector, sos_token_id=sos_token_id)  # (1, max_len)

# --- Reconstruye texto ---
decoded_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
print(f"🔓 Texto reconstruido: {decoded_text}")