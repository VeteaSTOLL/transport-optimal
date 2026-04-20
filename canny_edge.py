import cv2
import matplotlib.pyplot as plt

img = cv2.imread("JEUNE_JOL.jpg", cv2.IMREAD_GRAYSCALE)
#seuil petit -> plus de contour (plsu de bruit)
#seuilf grand -> inverse
edges = cv2.Canny(img, 50, 100)

plt.imshow(edges, cmap='gray')
plt.title("Contours Canny")
plt.axis('off')
plt.show()