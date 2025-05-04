import fiftyone.zoo as foz
import fiftyone.core.expressions as foe
import fiftyone as fo

N_MAX = 100000
# Descargar hasta N_MAX imágenes del split "train"
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    max_samples=N_MAX,
    only_matching=False,
    shuffle=True
)

# Campo correcto para las detecciones
detections_field = "ground_truth"

# ViewField para acceder a las etiquetas
label = foe.ViewField("label")

# Filtrar imágenes que tienen al menos una detección "Human face"
face_view = dataset.filter_labels(detections_field, label == "Human face")

# Crear vista de imágenes SIN caras excluyendo las que están en face_view
images_without_faces = dataset.exclude(face_view)

# También puedes obtener explícitamente las que sí tienen caras
images_with_faces = face_view

print(f"🔍 Imágenes con caras: {len(images_with_faces)}")
print(f"🔍 Imágenes sin caras: {len(images_without_faces)}")

# Exportar hasta 10k imágenes sin caras
subset_no_faces = images_without_faces.take(N_MAX)
subset_no_faces.export(
    export_dir="data/openimages_no_faces",
    dataset_type=fo.types.ImageDirectory,
    label_field=None
)

# Exportar hasta 10k imágenes con caras (opcional)
subset_with_faces = images_with_faces.take(30000)
subset_with_faces.export(
    export_dir="data/openimages_with_faces",
    dataset_type=fo.types.ImageDirectory,
    label_field=None
)

print("✅ Exportación completada.")
