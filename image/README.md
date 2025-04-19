# StenGAN

Este proyecto implementa un sistema de esteganografía basado en GANs donde un mensaje binario es ocultado dentro de una imagen y luego recuperado, mientras se intenta mantener la imagen resultante visualmente similar a la original. 

## Arquitectura general

- **Encoder**: toma una imagen y un mensaje binario, y produce una imagen modificada (stego image) que contiene el mensaje oculto.
- **Decoder**: intenta recuperar el mensaje binario original a partir de la stego image.
- **Discriminador**: intenta distinguir entre imágenes reales y stego images, ayudando al encoder a generar imágenes más realistas.

## Dataset
- CIFAR-10
- Resolución: 32x32 RGB
- Escalado: [-1, 1] con Normalize((0.5,), (0.5,))

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

- **Warm-up**: durante las primeras 50 épocas no se usa `adv_loss` (solo reconstrucción)
- **Label smoothing**: etiquetas reales = 0.9, falsas = 0.1 (solo para BCE; ya no aplica con LSGAN)
- **Discriminador adaptativo**: si el discriminador se vuelve demasiado preciso (`disc_loss < 0.1`), se pausa su entrenamiento
- **Entrenamiento intermitente del D**: solo se entrena cada 15 pasos después del warm-up

### Pérdidas registradas en TensorBoard
- `Loss/Image`, `Loss/Message`, `Loss/Discriminator`, `Loss/Adversarial`
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

