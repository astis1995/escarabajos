
import cv2
import numpy as np

def load_and_preprocess_image(filepath, k=10):
    """
    Carga una imagen en escala de grises, convierte a negro los pixeles del top k%,
    luego redimensiona a 512x512.

    Parámetros:
    - filepath: ruta del archivo de imagen
    - k: porcentaje superior de la imagen a convertir en negro (0-100)

    Retorna:
    - img_gray: imagen preprocesada (np.ndarray)
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {filepath}")

    height = img.shape[0]
    top_k_pixels = int((k / 100) * height)

    # Convertir a negro la parte superior k%
    img[:top_k_pixels, :] = 0

    # Ahora redimensionar
    img = cv2.resize(img, (512, 512))

    return img
    
import cv2
import numpy as np

def noise_reduction(img, kernel_size=3):
    """
    Aplica un filtro de mediana para reducir ruido en la imagen.

    ¿Cómo funciona?
    Para cada píxel, considera una grilla (ventana) de tamaño kernel_size x kernel_size,
    ordena los píxeles de esa grilla de menor a mayor, toma el valor mediano,
    y sustituye el píxel central por ese valor.

    Parámetros:
    - img: imagen en escala de grises (np.ndarray)
    - kernel_size: tamaño de la grilla (debe ser impar, por ejemplo, 3, 5, 7)

    Retorna:
    - img_denoised: imagen sin ruido (np.ndarray)
    """

    img_denoised = cv2.medianBlur(img, kernel_size)
    return img_denoised
