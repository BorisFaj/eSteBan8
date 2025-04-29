import chromadb
import os

# Conectar cliente
client = chromadb.PersistentClient(path="/home/bfajardo/PycharmProjects/PythonProject/DAna/image/chromadb_storage")

# Cargar colección
train_collection = client.get_collection(name="embeddings_train")

# Leer imágenes del directorio
image_dir = "/home/bfajardo/PycharmProjects/PythonProject/DAna/image/openimages_custom/train"
image_names = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(".jpg")
])

print(f"📂 {len(image_names)} imágenes encontradas en {image_dir}")

# Buscar cuáles de esas imágenes existen en la colección
found = []
not_found = []

batch_size = 100

for i in range(0, len(image_names), batch_size):
    batch = image_names[i:i+batch_size]
    try:
        results = train_collection.get(ids=batch)
    except Exception as e:
        print(f"[⚠️] Error en batch {i//batch_size}: {str(e)}")
        continue

    if results and "ids" in results:
        found.extend(results["ids"])
    missing_in_batch = set(batch) - set(results.get("ids", []))
    not_found.extend(missing_in_batch)

print(f"✅ {len(found)} imágenes encontradas en ChromaDB.")
print(f"🚫 {len(not_found)} imágenes NO encontradas.")

# Si quieres ver ejemplos
print("\nEjemplos de imágenes encontradas:")
print(found[:5])

print("\nEjemplos de imágenes NO encontradas:")
print(not_found[:5])
