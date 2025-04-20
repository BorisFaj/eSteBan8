# 🧠 TextAutoencoder-VAE

Este proyecto implementa un autoencoder de texto que **comprime frases usando DistilBERT (como VAE)** y las reconstruye usando un **decoder GRU con teacher forcing y regularización**. Es ideal para explorar compresión semántica de texto y generación autoregresiva.

---

## 🌍 Arquitectura general

```
Texto → DistilBERT → μ, logσ² → sampling → z → GRU Decoder → Texto reconstruido
```

- `TextCompressorVAE`: usa DistilBERT para codificar frases, devuelve `μ`, `logσ²` y aplica sampling para obtener `z`.
- `TextDecoder`: genera texto desde `z` usando una GRU autoregresiva con teacher forcing y dropout.
- Pérdida combinada: `CrossEntropy + KL Divergence`

---

## 🚀 Características destacadas

- 🔁 **VAE encoder** con sampling latente (reparametrización).
- ✍️ **GRU decoder autoregresivo** con teacher forcing y dropout.
- 📏 Evaluación con BLEU (con smoothing), ROUGE y Perplexity.
- 📊 Visualización completa en TensorBoard: reconstrucciones, métricas, pesos, embeddings latentes.

---

## 📁 Estructura del proyecto

```
.
├── text_coder.py        # Compresor VAE basado en DistilBERT
├── text_decoder.py      # Decoder GRU autoregresivo
├── train.py             # Entrenamiento completo
├── checkpoints/         # Modelos guardados periódicamente
└── runs/                # Logs para TensorBoard
```

---

## 📈 Métricas

- **BLEU (con smoothing)**: compara el texto original con el reconstruido.
- **ROUGE-1 / ROUGE-L**: medidas de superposición léxica.
- **Perplexity**: mide la incertidumbre del decoder.
- **KL Divergence**: regulariza el espacio latente para hacerlo más normalizado.

---

## 🧪 Dataset

- [`wikitext-2-raw-v1`](https://huggingface.co/datasets/wikitext)
- Se filtran frases entre 10 y 50 palabras.
- Tokenización con `distilbert-base-uncased`

---

## 📊 Visualización con TensorBoard

Lanza:
```bash
tensorboard --logdir runs
```
Y verás:
- Gráficas de loss, BLEU, ROUGE, Perplexity.
- Textos originales y reconstruidos por epoch.
- Embeddings del espacio latente.
- Histogramas de pesos del decoder.

---

## ⚙️ Requisitos

```bash
pip install torch transformers datasets evaluate nltk tensorboard
```

---

## 🚧 Futuras mejoras

* [ ] Atención explícita en el decoder
* [ ] Compresor VAE entrenable (descongelar BERT)
* [ ] Comparación con modelos como BART o T5
