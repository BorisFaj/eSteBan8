# eSteBan8

Este proyecto implementa un sistema de esteganografía basado en GANs donde un mensaje binario es ocultado dentro de una imagen y luego recuperado, mientras se intenta mantener la imagen resultante visualmente similar a la original.

## 🧱 Arquitectura general

- **Encoder**: toma una imagen y un mensaje binario, y produce una imagen modificada (stego image) que contiene el mensaje oculto.
  - Se han añadido **residual skip connections**, **LayerNorm** y **Kaiming Initialization** para mejorar la estabilidad y capacidad del modelo.
- **Decoder**: intenta recuperar el mensaje binario original a partir de la stego image.
  - Ahora incluye inicialización dinámica del `Linear`, LayerNorm, skip connections e inicialización explícita.
- **Discriminador**: intenta distinguir entre imágenes reales y stego images, ayudando al encoder a generar imágenes más realistas.

## 📦 Dataset

- Actualmente **no se utiliza CIFAR-10**. Se usa un dataset personalizado (revisar `data_handler.py`).
- Las imágenes se escalan a resolución `32x32` o la especificada por `IMAGE_INPUT_RES` y se normalizan a `[-1, 1]`.

## ⚙️ Entrenamiento

- `batch_size=1` por motivos de memoria (actualmente entrenando de uno en uno).
- `message_size = 512` bits ocultos por imagen (≈ 0.166 bpp).
- El encoder y decoder se entrenan para minimizar:
  - **SSIM loss** + **MSE**: preservación visual (`image_loss_lambda` controla la ponderación).
  - **Binary Cross Entropy** para el mensaje (`message_loss`).
  - **Adversarial loss**: Least Squares GAN (`MSE` en vez de `BCE`).
- **Técnicas modernas**:
  - Mixed Precision Training con `torch.cuda.amp`.
  - `GradScaler` para escalar automáticamente los gradientes.
  - Entrenamiento del discriminador adaptativo (pausado dinámico).
  - Guardado de modelos y métricas en **TensorBoard y MLflow**.
  - Histograma de pesos y gradientes en TensorBoard.

## 📊 Métricas registradas

- **TensorBoard**:
  - `Loss/*`, `Accuracy/Bit`, histogramas de pesos y gradientes, imágenes reales vs stego.
- **MLflow**:
  - Todos los valores anteriores + modelos registrados (`log_model`) y artefactos (`log_artifact`).

## 🤪 Evaluación periódica

Cada `EPOCHS_TO_VAL` se evalúa en el conjunto de test:
- Se mide: `message_loss`, `image_loss`, `bit_accuracy`, `style_loss`.
- Se loguea a TensorBoard y MLflow.
- Se generan visualizaciones de comparación de imágenes.

## 📌 Consideraciones adicionales

- El encoder ahora **modifica la imagen original directamente** para ocultar el mensaje.
- Los modelos pueden ser **guardados periódicamente** cada `EPOCHS_TO_SAVE` y subidos a MLflow.
- El `Decoder` ahora se inicializa dinámicamente según la forma de la salida convolucional, y está preparado para trabajar con imágenes de cualquier tamaño.

## ✅ Cosas pendientes / ToDo

- [ ] Mejorar la representación del mensaje en el espacio latente.
- [ ] Comprobar calidad visual en datasets más grandes (e.g. CelebA).
- [x] Explorar diferentes valores de `image_loss_lambda` (0.01, 0.1, 0.5...)
- [x] Evaluar la precisión del mensaje recuperado (`bit accuracy`)
- [x] Probar `Tanh` vs `Sigmoid` en la salida del encoder (ya aplicado: usando Tanh)
- [x] Escalar a imágenes más grandes o usar datasets más realistas (e.g. CelebA)
- [x] Añadir modo de inferencia: cargar un modelo y visualizar imagen + mensaje recuperado
- [ ] El decoder separa el mensaje y la imagen (ahora solo extrae el mensaje)
- [ ] Decoder pausado dinámicamente: si message_loss baja de cierto umbral (ej. 0.01), se puede detener su entrenamiento temporalmente para acelerar el aprendizaje del encoder y del discriminador
- [ ] Cuando el generador sea capaz de generar imagenes reales, evaluar el discriminador con la imagen SIN ruido

# Análisis de capacidad de ocultación en redes esteganográficas

## 🎯 Objetivo
Determinar cuán parecida puede ser una imagen modificada (stego) a la original al esconder un mensaje binario, sin que el discriminador logre distinguirla, y sin que el entrenamiento del generador (encoder) se frene.

## 📐 Parámetros relevantes

- **Tamaño del mensaje (`message_size`)**: cantidad de bits que se desean esconder.
- **Resolución de la imagen (`C x H x W`)**: espacio disponible para esconder.
- **`disc_loss`**: pérdida del discriminador al clasificar imágenes reales vs stego.
- **`SSIM + MSE`**: métricas de similitud visual usadas para evitar degradación visible.

## 📊 Curva empírica simulada

| Tamaño del mensaje (bits) | `disc_loss` promedio |
|---------------------------|-----------------------|
| 64                        | 0.02                  |
| 128                       | 0.04                  |
| 256                       | 0.09                  |
| 512                       | 0.17                  |
| 1024                      | 0.28                  |
| 2048                      | 0.42                  |

🔴 Umbral de detección: `disc_loss > 0.1`

> El sistema comienza a ser detectable con mensajes de más de ~300 bits.

## 🕰️ Tiempo vs Capacidad

- Aunque un tamaño mayor es posible, **el número de epochs para vencer al discriminador crece**.
- El tiempo de entrenamiento y los recursos disponibles se convierten en **restricciones adicionales**.

## 🧠 Observación clave

> Debes observar los histogramas del `encoder` en TensorBoard:
>
> - Si las activaciones o pesos dejan de cambiar, el `encoder` está estancado.
> - Asegúrate de que sigue aprendiendo una vez que el `discriminator` se ha debilitado.

## ✅ Recomendaciones

- Vigilar `disc_loss`, `message_loss` y `bit_accuracy`.
- Si `disc_loss < 0.1` y `bit_accuracy > 0.99`, el sistema está en zona segura.
- Considerar reducir el tamaño del mensaje o aumentar el número de parámetros solo si el `encoder` se estanca antes de converger.

