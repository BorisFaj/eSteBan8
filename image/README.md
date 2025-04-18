# GAN Steganography Prototype

Este módulo es un prototipo de sistema basado en GANs y autoencoders para ocultar mensajes dentro de imágenes.

## Componentes

- **Encoder**: recibe una imagen y un mensaje, y genera un vector latente `z`.
- **Decoder**: a partir de `z`, reconstruye la imagen original y el mensaje.
- **Discriminador**: intenta distinguir si `z` proviene del encoder o de una distribución aleatoria.

## Pérdidas

- `SSIM` para preservar la calidad visual de la imagen reconstruida.
- `BCE` para la reconstrucción del mensaje.
- `BCE adversarial` para entrenar el discriminador y el generador.

## ToDo

- [ ] Probar con `MSE` en lugar de `SSIM` como pérdida de imagen.
- [ ] Añadir entrenamientos por épocas y dataloader.
- [ ] Visualizar imágenes originales vs reconstruidas.
- [ ] Evaluar capacidad máxima de ocultación (bits por pixel).
