from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch
import chromadb
import os


def text_and_image_to_bert_chromadb(
        captions_path: str,
        image_dir: str,
        collection_name: str,
        image_names_path: str = None
):
    # Paso 1: Leer captions
    with open(captions_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Paso 2: Leer imágenes en orden controlado
    if image_names_path and os.path.exists(image_names_path):
        with open(image_names_path, "r", encoding="utf-8") as f:
            image_names = [line.strip() for line in f if line.strip()]
        print(f"📄 Cargando nombres de imagen desde '{image_names_path}'.")
    else:
        image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        if image_names_path:
            with open(image_names_path, "w", encoding="utf-8") as f:
                for name in image_names:
                    f.write(f"{name}\n")
            print(f"✅ Orden de imágenes guardado en '{image_names_path}'.")

    assert len(sentences) == len(image_names), f"⚠️ {len(sentences)} captions vs {len(image_names)} imágenes"

    # Paso 3: Preparar BERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Paso 4: Preparar ChromaDB persistente
    client = chromadb.PersistentClient(path="./chromadb_storage")

    # BORRAR la colección anterior si existe
    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️ Colección '{collection_name}' borrada antes de crearla de nuevo.")
    except:
        print(f"ℹ️ Colección '{collection_name}' no existía, creando nueva.")

    collection = client.create_collection(name=collection_name)

    # Paso 5: Procesar e insertar
    with torch.no_grad():
        for i in tqdm(range(len(sentences)), desc="Procesando texto + guardando en ChromaDB"):
            sentence = sentences[i]
            image_name = image_names[i]
            image_id = os.path.splitext(image_name)[0]

            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().float()

            collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "caption": sentence,
                    "image_name": image_name
                }],
                ids=[image_id]
            )

            del inputs, outputs
            torch.cuda.empty_cache()

    print(f"✅ {len(sentences)} embeddings guardados en colección '{collection_name}' en './chromadb_storage'.")


if __name__ == "__main__":
    text_and_image_to_bert_chromadb(
        captions_path="data/generated_train.txt",
        image_dir="openimages_custom/train",
        collection_name="embeddings_train",
        image_names_path="data/image_names_train.txt"
    )
