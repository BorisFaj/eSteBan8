from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import chromadb
import os

def text_and_image_to_bert_chromadb(
    captions_path: str,
    image_dir: str,
    collection_name: str):
    with open(captions_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    image_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    assert len(sentences) == len(image_names), f"⚠️ {len(sentences)} captions vs {len(image_names)} imágenes"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    client = chromadb.PersistentClient(path="./chromadb_storage")

    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️ Colección '{collection_name}' borrada antes de crearla de nuevo.")
    except:
        print(f"ℹ️ Colección '{collection_name}' no existía, creando nueva.")

    collection = client.create_collection(name=collection_name)

    with torch.no_grad():
        for sentence, image_name in tqdm(zip(sentences, image_names), total=len(sentences), desc="Procesando"):
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().float()

            # Insertar en ChromaDB usando nombre de imagen completo como ID
            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "caption": sentence,
                    "image_name": image_name
                }],
                ids=[image_name]  # Aquí guardamos el nombre COMPLETO de imagen como ID
            )

            del inputs, outputs
            torch.cuda.empty_cache()

    print(f"✅ {len(sentences)} embeddings guardados en colección '{collection_name}' en './chromadb_storage'.")

if __name__ == "__main__":
    text_and_image_to_bert_chromadb(
        captions_path="data/generated_train.txt",
        image_dir="openimages_custom/train",
        collection_name="embeddings_train"
    )
