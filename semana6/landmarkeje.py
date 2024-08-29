#%%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Carpeta donde se encuentran las imágenes
folder = 'images_land'

# Leer las imágenes en escala de grises desde la nueva ubicación
img1 = cv.imread(os.path.join(folder, 'land0.jpg'), cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread(os.path.join(folder, 'src0.jpg'), cv.IMREAD_GRAYSCALE)  # trainImage

# Iniciar el detector ORB
orb = cv.ORB_create()

# Encontrar los puntos clave y descriptores con ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Crear el objeto BFMatcher
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Coincidir descriptores
matches = bf.match(des1, des2)

# Ordenar las coincidencias por su distancia
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar las primeras 10 coincidencias
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Obtener las coordenadas de los puntos clave coincidentes
points = np.array([kp2[m.trainIdx].pt for m in matches[:10]]).astype(int)

# Calcular el rectángulo delimitador
x, y, w, h = cv.boundingRect(points)

# Dibujar el rectángulo en la imagen de entrenamiento
img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
cv.rectangle(img2_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen resultante con el rectángulo
plt.subplot(1, 2, 1), plt.imshow(img3), plt.title('Coincidencias')
plt.subplot(1, 2, 2), plt.imshow(img2_color), plt.title('Rectángulo')
plt.show()

# %%
