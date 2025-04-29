from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import os

class JPEGWithEmbeddingFromChromaDataset(Dataset):
    def __init__(self, image_dir, collection_name, vector_db_client, image_names_path, image_size=224, batch_size=100):
        # Cargar nombres de imagen en orden controlado
        with open(image_names_path, "r", encoding="utf-8") as f:
            image_names = [line.strip() for line in f if line.strip()]

        self.image_paths = [os.path.join(image_dir, img_name) for img_name in image_names]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.collection = vector_db_client.get_collection(name=collection_name)

        # Precargar embeddings en RAM
        self.embeddings = {}
        all_ids = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]
        missing_ids = []

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            try:
                results = self.collection.get(ids=batch_ids)
            except Exception as e:
                print(f"[⚠️] Error en batch {i//batch_size}: {str(e)}")
                continue

            if results is None or not results.get("ids") or not results.get("embeddings"):
                print(f"[⚠️] No se encontraron embeddings para batch con IDs: {batch_ids[:5]}...")
                missing_ids.extend(batch_ids)
                continue

            for id_, embedding in zip(results["ids"], results["embeddings"]):
                if embedding is not None:
                    self.embeddings[id_] = torch.tensor(embedding, dtype=torch.float32)
                else:
                    missing_ids.append(id_)

        print(f"✅ Precargados {len(self.embeddings)} embeddings de {len(all_ids)} posibles.")
        if missing_ids:
            print(f"🚫 Faltan {len(missing_ids)} embeddings.")
            print(f"🔍 Ejemplo de IDs faltantes: {missing_ids[:5]}")

        # Filtrar imágenes que sí tienen embedding
        self.image_paths = [
            p for p in self.image_paths
            if os.path.splitext(os.path.basename(p))[0] in self.embeddings
        ]

        print(f"📦 Dataset final contiene {len(self.image_paths)} imágenes con embeddings.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).float()

        embedding = self.embeddings[image_id]
        return image, embedding
