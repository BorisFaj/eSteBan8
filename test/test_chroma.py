from chromadb import PersistentClient

client = PersistentClient(path="/home/boris/PycharmProjects/eSteBan8/image/chromadb_storage")
collection = client.get_collection(name="embeddings_train")

print("🧠 Colección cargada:", collection.name)
print("📦 Número de elementos:", collection.count())

# Muestra algunos IDs
peek = collection.peek(5)
print("🔍 Ejemplo de IDs en ChromaDB:")
for i, id_ in enumerate(peek["ids"]):
    print(f"  {i+1}. {id_}")
