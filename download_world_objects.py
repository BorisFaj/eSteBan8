import fiftyone.zoo as foz
import fiftyone.core.expressions as foe
import fiftyone as fo

import face_recognition
from tqdm import tqdm
import os
import shutil

# -------------------------
# 1. Descargar COCO y filtrar por persona
# -------------------------

N_MAX = 10000

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    max_samples=N_MAX,
    shuffle=True,
)
print("🧪 Mostrando campos del primer sample del dataset:")
print(dataset.first())

detections_field = "ground_truth"
label = foe.ViewField("label")

images_with_person = dataset.filter_labels(detections_field, label == "person")
images_without_person = dataset.exclude(images_with_person)

print(f"👤 Con personas: {len(images_with_person)}")
print(f"🚫 Sin personas (candidatas a sin cara): {len(images_without_person)}")

# -------------------------
# 2. Usar face_recognition para detectar caras
# -------------------------

print("🔍 Detectando caras con face_recognition...")
no_face_samples = []

for sample in tqdm(images_without_person):
    path = sample.filepath
    try:
        image = face_recognition.load_image_file(path)
        faces = face_recognition.face_locations(image)

        if len(faces) == 0:
            no_face_samples.append(sample)
    except Exception as e:
        print(f"❌ Error procesando {path}: {e}")

print(f"✅ Imágenes sin personas y sin caras detectadas: {len(no_face_samples)}")

# -------------------------
# 3. Exportar imágenes sin caras detectadas
# -------------------------

EXPORT_DIR = "data/clean_no_faces"
os.makedirs(EXPORT_DIR, exist_ok=True)

for sample in tqdm(no_face_samples):
    src = sample.filepath
    dst = os.path.join(EXPORT_DIR, os.path.basename(src))
    shutil.copyfile(src, dst)

print(f"📦 Exportadas {len(no_face_samples)} imágenes a: {EXPORT_DIR}")
