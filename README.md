# eSteBan8

**eSteBan8** es un proyecto de investigación y desarrollo en esteganografía y recuperación de mensajes ocultos mediante Deep Learning. El sistema se compone de dos grandes bloques:

- **GAN de imágenes para esteganografía visual** (`image/`).
- **LSTM para recuperación de embeddings de texto (BERT)** (`message/`).

---

## ✨ Módulo 1: GAN de imágenes (carpeta `image/`)

Este módulo implementa una arquitectura GAN adaptada para esteganografía:

- El **Encoder** inserta un mensaje en una imagen minimizando la alteración visual.
- El **Decoder** recupera el mensaje de la imagen stego.
- El **Discriminator** intenta diferenciar entre imágenes reales y stego.

### Principales hiperparámetros

| Nombre | Valor | Descripción |
|:---|:---|:---|
| `WARM_UP_LEN` | `10` | Epochs iniciales donde no se entrena el discriminador. |
| `image_loss_lambda` | `0.25` | Peso de la pérdida de imagen (MSE respecto a SSIM). |
| `FREEZE_DISC_LOSS` | `0.05` | Umbral para congelar el discriminador (personalizable). |
| `image_channels` | `3` | Canales de la imagen (RGB). |
| `image_size` | `128` | Tamaño de las imágenes cuadradas (128x128 px). |
| `message_size` | `768` | Dimensión del mensaje embebido. |
| `batch_size` | `64` | Tamaño del batch de entrenamiento. |
| `num_epochs` | `3000` | Número total de épocas de entrenamiento. |
| `IMAGE_INPUT_RES` | `128` | Resolución de entrada. |
| `EPOCHS_TO_VAL` | `5` | Frecuencia de evaluación. |
| `EPOCHS_TO_SAVE` | `10` | Frecuencia de guardado de checkpoints. |
| `noise_std` | `0.01` | Nivel de ruido gaussiano opcional (actualmente desactivado). |
| `style_loss_weight` | `0.1` | Peso para una pérdida de estilo (reservado). |
| `message_weight` | `1.0` | Peso de la pérdida del mensaje. |
| `disc_loss_target` | `0.1` | Pérdida objetivo del discriminador. |
| `message_loss_target` | `0.1` | Pérdida objetivo del mensaje. |
| `sharpness` | `0.3` | Controla la suavidad en la activación del discriminador. |
| `message_alpha` | `0.001` | Peso de regularización L2 sobre el mensaje. |

---

## 🧠 Módulo 2: Recuperación de embeddings (carpeta `message/`)

Este módulo entrena una red basada en LSTM para reconstruir embeddings de texto (por ejemplo, vectores BERT) a partir de datos de entrada:

- Entrada: secuencias de imágenes modificadas o embeddings parciales.
- Salida: embeddings de 768 dimensiones.

### Objetivos
- Minimizar la distancia entre embeddings originales y recuperados.
- Mantener la consistencia semántica del mensaje.

### Futuras extensiones
- Uso de Transformers en lugar de LSTM.
- Fine-tuning de embeddings.
- Regularizaciones avanzadas.

---

## 📂 Estructura general del proyecto

```
eSteBan8/
├── image/                # GAN de imágenes
│   ├── encoder.py
│   ├── decoder.py
│   ├── discriminator.py
│   ├── train.py
│   └── ...
├── message/              # LSTM para recuperación de embeddings
│   ├── lstm_recover.py
│   ├── train.py
│   └── ...
├── checkpoints/
├── logs/
├── runs/
├── .env
├── README.md
└── requirements.txt
```

---

## 💪 Herramientas

- **PyTorch**: framework principal de entrenamiento.
- **TensorBoard**: visualización de métricas e imágenes.
- **MLflow**: gestión de experimentos y logging avanzado.

---

## 🚀 Primeros pasos

```bash
# Clonar el repositorio
git clone git@github.com:BorisFaj/eSteBan8.git
cd eSteBan8

# Instalar dependencias
pip install -r requirements.txt

# Configurar tu archivo .env correctamente

# Entrenar el modelo de imágenes
cd image
python train.py

# Entrenar el modelo de recuperación de mensajes
cd ../message
python train.py
```

---

## 📜 Licencia

Este proyecto está licenciado bajo la **Licencia Apache 2.0**.

Consulta el archivo [LICENSE](LICENSE) para más detalles.
