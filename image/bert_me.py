from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import os

def text_and_image_to_bert(
    captions_path: str,
    output_dir: str,
):
    with open(captions_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(len(sentences)), desc="Procesando imágenes + texto"):
            sentence = sentences[i]

            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().half()  # float16

            torch.save(embedding, os.path.join(output_dir, f"{i:06d}.pt"))

            # Liberar explícitamente memoria de GPU
            del inputs, outputs
            torch.cuda.empty_cache()

    print(f"✅ {len(sentences)} muestras guardadas en {output_dir}")

if __name__ == "__main__":
    text_and_image_to_bert(
        captions_path="data/generated_val.txt",
        output_dir="data/val_preprocessed"
    )
