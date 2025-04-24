from transformers import pipeline
import torch
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

def clean_phrase(text, prompt):
    return text.replace(prompt, "").strip().replace("\n", " ")

def gen_file(path="data"):
    train_file = os.path.join(path, "generated_train.txt")
    val_file = os.path.join(path, "generated_val.txt")

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline('text-generation', model='heegyu/gpt2-emotion', device=device)

    emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    prompts = [f"I feel {emotion} because" for emotion in emotions]

    train_count, val_count = get_total_images()
    total_needed = train_count + val_count
    generated_phrases = set()

    print("🧠 Generando frases...")
    while len(generated_phrases) < total_needed:
        prompt = random.choice(prompts)
        result = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        phrase = clean_phrase(result, prompt)

        if is_valid_phrase(phrase):
            generated_phrases.add(phrase)

        if len(generated_phrases) % 50 == 0:
            print(f"✔️ {len(generated_phrases)} frases válidas")

    phrases = list(generated_phrases)
    random.shuffle(phrases)

    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in phrases[:train_count])

    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in phrases[train_count:])

    print("✅ Frases generadas y guardadas")

if __name__ == "__main__":
    gen_file()
