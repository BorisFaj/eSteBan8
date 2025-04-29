from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import chromadb
import os

def text_and_image_to_bert_chromadb(
    captions_path: str,
    image_dir: str,
    collection_name: str
):
    # Paso 1: Leer frases
    with open(captions_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Paso 2: Listar imágenes y asegurarse de que coinciden en número
    image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    assert len(sentences) == len(image_names), f"⚠️ {len(sentences)} captions vs {len(image_names)} imágenes"

    # Paso 3: Preparar BERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Paso 4: Preparar ChromaDB persistente
    client = chromadb.PersistentClient(path="./chromadb_storage")
    collection = client.get_or_create_collection(name=collection_name)

    # Paso 5: Procesar e insertar
    with torch.no_grad():
        for i in tqdm(range(len(sentences)), desc="Procesando texto + guardando en ChromaDB"):
            sentence = sentences[i]
            image_name = image_names[i]
            image_id = os.path.splitext(image_name)[0]  # "000001"

            # Obtener embedding
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().float()

            # Insertar en ChromaDB
            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "caption": sentence,
                    "image_name": image_name
                }],
                ids=[image_id]
            )

            # Limpiar memoria
            del inputs, outputs
            torch.cuda.empty_cache()

    print(f"✅ {len(sentences)} embeddings guardados en colección '{collection_name}' en ./chromadb_storage")

if __name__ == "__main__":
    text_and_image_to_bert_chromadb(
        captions_path="data/generated_val.txt",
        image_dir="openimages_custom/val",
        collection_name="embeddings_val"
    )
