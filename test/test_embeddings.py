import chromadb
import os

client = chromadb.PersistentClient(path="/home/boris/PycharmProjects/eSteBan8/image//chromadb_storage")
collection = client.get_collection(name="embeddings_train")
print("Ejemplo de IDs:", collection.peek(5)['ids'])
print("Total:", collection.count())
print([os.path.splitext(f)[0] for f in os.listdir("/home/boris/PycharmProjects/eSteBan8/image/openimages_custom/train") if f.endswith(".jpg")][:5])
