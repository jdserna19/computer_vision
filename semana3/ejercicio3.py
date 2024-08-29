#%%
#Punto 1
import cv2
import numpy as np

# Cargar la imagen
image_path = "cat.jpg"
image = cv2.imread(image_path)

# Convertir la imagen a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir el rango de colores para el fondo verde
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Crear una máscara para el fondo verde
mask = cv2.inRange(hsv, lower_green, upper_green)

# Invertir la máscara para obtener el gato
mask_inv = cv2.bitwise_not(mask)

# Crear una imagen de fondo del nuevo color (por ejemplo, azul)
new_background_color = [255, 0, 0]  # BGR para azul
background = np.full_like(image, new_background_color)

# Aplicar la máscara para extraer el fondo y el gato
background_masked = cv2.bitwise_and(background, background, mask=mask)
cat_masked = cv2.bitwise_and(image, image, mask=mask_inv)

# Combinar el gato con el nuevo fondo
final_image = cv2.add(background_masked, cat_masked)

# Guardar la imagen resultante
cv2.imwrite("/mnt/data/cat_with_new_background.jpg", final_image)

# Mostrar la imagen resultante
cv2.imshow("Cat with New Background", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#Punto 2A
import cv2
import numpy as np

def adjust_brightness_contrast(image, alpha, beta):
    # Convertir la imagen a tipo float32
    new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return new_image

# Cargar la imagen
image_path = "cat.jpg"
image = cv2.imread(image_path)

# Ajustar el brillo y el contraste
alpha = 1.5  # Factor de contraste
beta = 50    # Valor de brillo
adjusted_image = adjust_brightness_contrast(image, alpha, beta)

# Guardar y mostrar la imagen resultante
cv2.imwrite("/mnt/data/cat_brightness_contrast.jpg", adjusted_image)
cv2.imshow("Adjusted Brightness and Contrast", adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#Punto 2B
import cv2
import numpy as np

# Cargar la imagen
image_path = "cat.jpg"
image = cv2.imread(image_path)

# Convertir la imagen a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir el rango de colores para el fondo verde
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Crear una máscara para el fondo verde
mask = cv2.inRange(hsv, lower_green, upper_green)

# Invertir la máscara para obtener el gato
mask_inv = cv2.bitwise_not(mask)

# Convertir la máscara invertida a 3 canales
mask_inv_3channel = cv2.merge([mask_inv, mask_inv, mask_inv])

# Aplicar la máscara para obtener solo el gato
cat = cv2.bitwise_and(image, mask_inv_3channel)

# Guardar el gato como PNG con fondo transparente
output_path = "/mnt/data/cat_matted.png"
alpha_channel = mask_inv
cat_rgba = cv2.merge([cat[:, :, 0], cat[:, :, 1], cat[:, :, 2], alpha_channel])
cv2.imwrite(output_path, cat_rgba)

# Mostrar la imagen resultante
cv2.imshow("Cat Matting", cat_rgba)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
#Punto 3
import cv2
import numpy as np

def apply_sobel_operator(image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el operador Sobel en la dirección X
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Aplicar el operador Sobel en la dirección Y
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcular la magnitud del gradiente
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalizar la imagen resultante
    sobel_combined = np.uint8(np.absolute(sobel_combined))
    
    return sobel_combined

# Cargar la imagen
image_path = "cat.jpg"
image = cv2.imread(image_path)

# Aplicar el operador Sobel
sobel_image = apply_sobel_operator(image)

# Guardar y mostrar la imagen resultante
cv2.imwrite("/mnt/data/cat_sobel.jpg", sobel_image)
cv2.imshow("Sobel Operator", sobel_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
