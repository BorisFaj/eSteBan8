# eSteBan8

Sistema de esteganografía basado en GANs donde un mensaje binario es ocultado dentro de una imagen y luego recuperado, manteniendo alta calidad visual.

## 🧱 Arquitectura general

- **Encoder**:  
  - Recibe una imagen y un mensaje binario.
  - Produce una imagen stego (`stego image`) conteniendo el mensaje.
  - Incluye: **residual skip connections**, **LayerNorm**, **Kaiming Initialization**, **LeakyReLU**.
- **Decoder**:  
  - Recupera el mensaje a partir de la stego image.
  - Robusto: convoluciones, saltos residuales y `AdaptiveAvgPool` para trabajar con distintas resoluciones.
- **Discriminador**:  
  - Distingue imágenes reales de stego images.
  - Entrenamiento adaptativo en función del progreso del encoder (basado en `message_loss`).

## 📦 Dataset

- Dataset personalizado (`data_handler.py`).
- Imágenes reescaladas a `128x128` (`IMAGE_INPUT_RES=128`).
- Normalización en rango `[-1, 1]`.
- **Embeddings de mensaje** generados dinámicamente:
  - Se utiliza un modelo de texto **GPT-2 Emotion** (`heegyu/gpt2-emotion`) para generar frases aleatorias asociadas a emociones.
  - Las frases generadas se **tokenizan** y **se embeben** usando **BERT** (`bert-base-uncased`).
  - Los vectores resultantes (`message_size=768`) son usados como el mensaje binario a ocultar en las imágenes.
- El input completo para el Encoder es: `(imagen real, embedding textual)`.

## ⚙️ Entrenamiento

### Técnicas empleadas:

- **Mixed Precision Training**:  
  - Uso de `torch.cuda.amp` para reducir consumo de memoria y acelerar el entrenamiento.
  - `GradScaler` para evitar underflow de gradientes.
- **Warm-Up** inicial (`WARM_UP_LEN=30`):  
  - Durante las primeras epochs, el discriminador está congelado para dejar que encoder y decoder aprendan representación básica.
- **Disc Freeze Dinámico**:  
  - Se congela el discriminador cada vez que `disc_loss` baja de `FREEZE_DISC_LOSS=0.3`, durante una ventana de `DISC_FREEZE_WINDOW=15` epochs.
  - **Entrenamiento adaptativo** basado en performance, no en pasos fijos (curriculum dinámico).
- **Ruido progresivo** (pendiente de ampliar):  
  - Actualmente se añade ruido `noise_std=0.02` a las imágenes.
  - Se planea incrementar el ruido conforme baje el `message_loss` para hacer el sistema más robusto.
- **Logging exhaustivo**:  
  - **TensorBoard**: losses, precisión de bits, histogramas de pesos y gradientes, comparativas de imágenes reales vs stego.
  - **MLflow**: todos los logs + checkpoints de modelos + artefactos.

## 📊 Métricas registradas

- `Loss/message_loss`
- `Loss/image_loss`
- `Loss/disc_loss`
- `Accuracy/bit_accuracy`
- `SSIM`
- Histogramas de activaciones y pesos
- Comparativas de imágenes reales vs stego en TensorBoard.

## 📈 Conclusiones actuales

- Con arquitectura **Encoder+Decoder clásica** (sin discriminador), se requiere más profundidad para recuperar bien el mensaje.
- Con arquitectura **GAN + Discriminador adaptativo**, se obtiene:
  - Mucho mejor trade-off entre calidad visual y recuperación del mensaje.
  - Aprendizaje más rápido si se gestiona dinámicamente el entrenamiento del discriminador.
- Entrenar el discriminador sólo cuando el encoder alcanza un mínimo nivel de calidad (bajo `message_loss`) acelera la convergencia y evita inestabilidad.
- El sistema soporta correctamente mensajes de 768 bits embebidos en imágenes de 128x128 (~0.046 bpp).

## 🚀 Próximos pasos

- [ ] **Ruido progresivo**:  
  - Incrementar dinámicamente el `noise_std` conforme el `message_loss` disminuya, para forzar al sistema a ser más robusto.
- [ ] **Dataset más grande**:  
  - Entrenar sobre CelebA-HQ o similar para validar escalabilidad a imágenes de mayor resolución.
- [ ] **Pausa de módulos**:  
  - Implementar pausado automático del decoder o discriminador si el sistema alcanza suficiente precisión (`bit_accuracy > 0.999`, `disc_loss < 0.1`).
- [ ] **Ajustar `image_loss_lambda` y `message_weight`**:
  - Testear nuevos pesos para lograr mejor balance entre calidad visual y recuperación.
- [ ] **Implementar inferencia**:
  - Script para cargar un modelo entrenado, ocultar mensajes y recuperarlos en imágenes nuevas.
