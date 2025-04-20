# eSteBan8

Este proyecto implementa un sistema de esteganografía basado en GANs donde un mensaje binario es ocultado dentro de una imagen y luego recuperado, mientras se intenta mantener la imagen resultante visualmente similar a la original.

## Arquitectura general

- **Encoder**: toma una imagen y un mensaje binario, y produce una imagen modificada (stego image) que contiene el mensaje oculto.
- **Decoder**: intenta recuperar el mensaje binario original a partir de la stego image.
- **Discriminador**: intenta distinguir entre imágenes reales y stego images, ayudando al encoder a generar imágenes más realistas.

## Dataset

- CIFAR-10
- Resolución: 32x32 RGB
- Escalado: [-1, 1] con Normalize((0.5,), (0.5,))
- Conjunto de **entrenamiento**: 50,000 imágenes
- Conjunto de **test**: 10,000 imágenes (usado para validación)

## Entrenamiento

- Se entrena con batches de 64 imágenes.
- `message_size = 512` bits ocultos por imagen (≈ 0.166 bpp)
- El encoder y decoder se entrenan para minimizar:
  - **SSIM loss**: `1 - SSIM` para mantener la similitud estructural
  - **MSE**: ponderada por `image_loss_lambda = 0.1` para ayudar a estabilizar
  - **Message loss**: Binary Cross Entropy entre el mensaje original y el recuperado
    - Se aplica sobre vectores de bits (`[batch_size, message_size]`) y se promedia por batch.
  - **Adversarial loss**: ahora implementada con **Least Squares GAN (LSGAN)**
    - El generador minimiza `MSE(D(stego), 1)`
    - El discriminador minimiza `MSE(D(real), 1) + MSE(D(fake), 0)`
    - También aplicada sobre vectores de tamaño `[batch_size, 1]` y promediada.

### Técnicas usadas

- **Warm-up**: durante las primeras 100 épocas no se usa `adv_loss` (solo reconstrucción)
- **Label smoothing**: etiquetas reales = 0.9, falsas = 0.1 (solo para BCE; ya no aplica con LSGAN)
- **Discriminador adaptativo**: si el discriminador se vuelve demasiado preciso (`disc_loss < 0.1`), se pausa su entrenamiento
- **Entrenamiento intermitente del D**: solo se entrena cada 5 pasos después del warm-up
- **Decoder pausado dinámicamente**: si `message_loss` baja de cierto umbral (ej. 0.01), se puede detener su entrenamiento temporalmente para acelerar el aprendizaje del encoder y del discriminador
- **Evaluación en conjunto de test** cada 100 épocas:
  - Se calcula `bit accuracy`, `message loss` y `image loss` sobre datos no vistos
  - Se registran en TensorBoard como `Test/Accuracy/Bit`, `Test/Loss/Message`, etc.
  - Se visualizan imágenes stego vs reales del test

### Consideraciones sobre bit accuracy y límite teórico

- El `bit accuracy` mide la proporción de bits correctamente recuperados por el decoder.
- Aunque el número de bits por imagen (`message_size`) es fijo, el `bit accuracy` **es una métrica variable** y refleja la capacidad efectiva de codificación del sistema.
- En teoría, si la red tiene suficiente capacidad y el mensaje es pequeño en comparación al bpp disponible, el `bit accuracy` **puede alcanzar el 100%** (o `message_loss → 0`).
- Sin embargo, en la práctica, con `message_size = 512`, podrían observarse errores residuales debidos a:
  - Limitaciones en la capacidad del encoder
  - Ambigüedad o ruido en el proceso de codificación/decodificación
  - Interferencia del discriminador o presión de la adversarial loss
- Comparar el `bit accuracy` de `message_size = 128` frente a `512` puede revelar ese límite práctico: a menor `message_size`, suele mejorar el `bit accuracy`.

### Pérdidas registradas en TensorBoard

- `Loss/Image`, `Loss/Message`, `Loss/Discriminator`, `Loss/Adversarial`
- `Accuracy/Bit` durante entrenamiento
- `Test/Loss/*` y `Test/Accuracy/Bit` cada 100 épocas
- Histogramas de pesos y gradientes de todos los modelos
- Imágenes reales vs stego images por época
- `bpp` (bits per pixel) fijo, útil para comparar runs

## Cosas pendientes / ToDo

- [ ] Explorar diferentes valores de `image_loss_lambda` (0.01, 0.1, 0.5...)
- [ ] Comparar resultados visuales y métricas por valor de lambda
- [ ] Evaluar la precisión del mensaje recuperado (`bit accuracy`)
- [x] Probar `Tanh` vs `Sigmoid` en la salida del encoder (ya aplicado: usando Tanh)
- [ ] Escalar a imágenes más grandes o usar datasets más realistas (e.g. CelebA)
- [ ] Añadir modo de inferencia: cargar un modelo y visualizar imagen + mensaje recuperado
- [ ] El decoder separa el mensaje y la imagen (ahora solo extrae el mensaje)
- [ ] Decoder pausado dinámicamente: si message_loss baja de cierto umbral (ej. 0.01), se puede detener su entrenamiento temporalmente para acelerar el aprendizaje del encoder y del discriminador

