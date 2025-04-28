from datasets import load_dataset
import random
import os
import re

def count_images_in_folder(folder):
    return sum(len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
               for _, _, files in os.walk(folder))

def get_total_images(train_path="openimages_custom/train", val_path="openimages_custom/val"):
    train = count_images_in_folder(train_path)
    val = count_images_in_folder(val_path)
    print(f"Train: {train}, Val: {val}, Total: {train + val}")
    return train, val

def is_valid_phrase(text):
    if len(text.split()) < 5:
        return False
    if any(bad in text.lower() for bad in ["icky", "@", "#", "http", "lol", "hmm", "uh", "ah", "idk"]):
        return False
    if not re.match(r"^[A-Z]", text):  # debe empezar por mayúscula
        return False
    if not text.endswith(".") and not text.endswith("!"):
        return False
    return True

def clean_phrase(text):
    return text.strip().replace("\n", " ")

def gen_file(path="data"):
    train_file = os.path.join(path, "generated_train.txt")
    val_file = os.path.join(path, "generated_val.txt")

    train_count, val_count = get_total_images()
    total_needed = train_count + val_count

    print("🧠 Cargando dataset PAWS...")
    dataset = load_dataset("paws", "labeled_final", split="train")

    all_phrases = set()

    for entry in dataset:
        all_phrases.add(clean_phrase(entry['sentence1']))
        all_phrases.add(clean_phrase(entry['sentence2']))

    all_phrases = [p for p in all_phrases if is_valid_phrase(p)]

    print(f"✔️ {len(all_phrases)} frases válidas disponibles")

    if len(all_phrases) < total_needed:
        raise ValueError(f"No hay suficientes frases válidas ({len(all_phrases)}) para cubrir {total_needed} imágenes.")

    random.shuffle(all_phrases)

    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in all_phrases[:train_count])

    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in all_phrases[train_count:train_count + val_count])

    print("✅ Frases de PAWS descargadas y guardadas")

if __name__ == "__main__":
    gen_file()
