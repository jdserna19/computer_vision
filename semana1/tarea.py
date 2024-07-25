#%%
import matplotlib.pyplot as plt
import numpy as np
#%%
import cv2
#%%
from PIL import Image, ImageDraw
# %%
im = np.zeros((100,100), np.uint8)

# %%
plt.imshow(im, cmap='gray')
plt.axis('off')  # Ocultar los ejes
plt.show()
# %%
# Trazar una línea diagonal
for i in range(100):
    im[i, i] = 255  # Establecer el valor del píxel en blanco (255)

# Mostrar la imagen
plt.imshow(im, cmap='gray')
plt.axis('off')  # Ocultar los ejes
plt.show()
# %%
# Crear una imagen negra
im = np.zeros((100, 100, 3), dtype=np.uint8)

# Definir colores en formato RGB
black = [0, 0, 0]
green = [0, 255, 0]
red = [255, 0, 0]

# Función para dibujar un recuadro
def draw_square(image, center, size, color):
    x, y = center
    half_size = size // 2
    image[x-half_size:x+half_size+1, y-half_size:y+half_size+1] = color

# Centro de la imagen
center = (50, 50)

# Dibujar recuadros
draw_square(im, center, 10, black)
draw_square(im, center, 20, green)
draw_square(im, center, 30, red)

# Mostrar la imagen
plt.imshow(im)
plt.axis('off')  # Ocultar los ejes
plt.show()
# %%
# Crear una imagen negra
im = np.zeros((100, 100, 3), dtype=np.uint8)

# Definir colores en formato RGB
black = [0, 0, 0]
green = [0, 255, 0]
red = [255, 0, 0]
blue = [0, 0, 255]

# Función para dibujar un recuadro
def draw_square(image, center, size, color):
    x, y = center
    half_size = size // 2
    image[x-half_size:x+half_size+1, y-half_size:y+half_size+1] = color

# Centro de la imagen
center = (50, 50)

# Dibujar recuadros desde el más grande hasta el más pequeño
draw_square(im, center, 30, red)
draw_square(im, center, 20, green)
draw_square(im, center, 10, blue)
draw_square(im, center, 5, black)


# Mostrar la imagen
plt.imshow(im)
plt.axis('off')  # Ocultar los ejes
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# Crear una imagen negra
im = np.zeros((100, 100, 3), dtype=np.uint8)

# Generar un patrón de onda senoidal de izquierda a derecha
for i in range(100):
    for j in range(100):
        wave = int((np.sin(j / 5.0) * 127 + 128))
        im[i, j] = [wave * 0.5, 0, wave]  # Color morado

# Mostrar la imagen
plt.imshow(im)
plt.axis('off')  # Ocultar los ejes
plt.show()

# %%
