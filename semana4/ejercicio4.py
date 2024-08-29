#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image_path = "frutas.webp"
image = Image.open(image_path)
image_np = np.array(image)

#%%
# Reshape the image to a 2D array of pixels and 3 color values (RGB)
pixels = image_np.reshape(-1, 3)

# Apply K-means clustering to segment the image
kmeans = KMeans(n_clusters=5, random_state=42)  # 5 clusters for simplicity
kmeans.fit(pixels)

# Replace each pixel value with its corresponding centroid value
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image_np.shape)

# Convert to uint8 to properly display the image
segmented_img = segmented_img.astype(np.uint8)

# Plot the original and segmented images
plt.figure(figsize=(12, 6))
#%%
# Original image
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title('Original Image')
plt.axis('off')
#%%
# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title('Segmented Image with K-Means')
plt.axis('off')

plt.show()

# %%
import cv2

# Convert the image to float32 type as required by OpenCV's kmeans
pixels_cv = np.float32(pixels)
#%%
# Define criteria for the algorithm (type of termination, max iterations, required accuracy)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Apply kmeans() from OpenCV
k = 5  # Number of clusters
_, labels, centers = cv2.kmeans(pixels_cv, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#%%
# Convert centers back to uint8 (0-255 range) and map labels to corresponding colors
centers = np.uint8(centers)
segmented_img_cv = centers[labels.flatten()]
segmented_img_cv = segmented_img_cv.reshape(image_np.shape)
#%%
# Plot the results: original,  KMeans from OpenCV
plt.figure(figsize=(18, 6))
#%%
# Original image
plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title('Original Image')
plt.axis('off')

#%%
# OpenCV K-Means Segmented Image
plt.subplot(1, 3, 3)
plt.imshow(segmented_img_cv)
plt.title('Segmented Image (OpenCV)')
plt.axis('off')

plt.show()

# %%
# Ajustar el tamaño de las imágenes para que se vean más grandes
plt.figure(figsize=(24, 8))

# Imagen original
plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title('Original Image')
plt.axis('off')

# Imagen segmentada con Sklearn
plt.subplot(1, 3, 2)
plt.imshow(segmented_img)
plt.title('Segmented Image (Sklearn)')
plt.axis('off')

# Imagen segmentada con OpenCV
plt.subplot(1, 3, 3)
plt.imshow(segmented_img_cv)
plt.title('Segmented Image (OpenCV)')
plt.axis('off')

plt.show()

# %%


# %%
