import torch
from transformers import AutoTokenizer, AutoModel
from transformer_decoder import TransformerDecoder
import torch.nn.functional as F


def decode_text(input_text: str, tokenizer, bert, decoder):
    sos_token_id = TOKENIZER.cls_token_id
    pad_token_id = TOKENIZER.pad_token_id
    input_ids = tokenizer.encode(input_text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    attention_mask = (input_ids != pad_token_id).long()

    with torch.no_grad():
        memory = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        output_logits = decoder(memory, generate=True, sos_token_id=sos_token_id)
        output_ids = torch.argmax(output_logits, dim=-1)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return decoded

def bert_similarity(text1: str, text2: str, tokenizer, bert):
    # Tokenización
    ids1 = tokenizer(text1, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN).to(DEVICE)
    ids2 = tokenizer(text2, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN).to(DEVICE)

    # Embeddings BERT
    with torch.no_grad():
        z1 = bert(**ids1).last_hidden_state[:, 0, :]  # [CLS]
        z2 = bert(**ids2).last_hidden_state[:, 0, :]

    # Similitud coseno
    cos_sim = F.cosine_similarity(z1, z2).item()
    l2_dist = torch.norm(z1 - z2).item()

    return cos_sim, l2_dist


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # --- Configuración ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_LEN = 60
    CKPT_PATH = './models/TransBert_v0.1_epoch80.pt'

    # --- Cargar tokenizer y modelos ---
    TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = TOKENIZER.vocab_size

    _bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    _decoder = TransformerDecoder(embedding_dim=768, vocab_size=vocab_size, max_len=MAX_LEN).to(DEVICE)

    # --- Cargar checkpoint ---
    _checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    _decoder.load_state_dict(_checkpoint['decoder_state_dict'])
    _decoder.eval()
    _bert.eval()

    _input_text = "Machine learning models require a lot of data"
    _output_text = decode_text(_input_text, TOKENIZER, _bert, _decoder)

    print(f"📝 Input:  {_input_text}")
    print(f"🔁 Output: {_output_text}")

    cos, l2 = bert_similarity(_input_text, _output_text, TOKENIZER, _bert)
    print(f"🔗 Cosine similarity: {cos:.4f}")
    print(f"📏 L2 distance:      {l2:.4f}")

