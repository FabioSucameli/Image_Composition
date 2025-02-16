import os
import cv2
import numpy as np
from glob import glob
from joblib import Parallel, delayed


def load_images(input_dir, max_images):
    image_paths = glob(os.path.join(input_dir, "*.png"))[:max_images]
    images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths]
    return images, image_paths

# Sovrappone un'immagine overlay (ridotta del 50% con canale alfa per la trasparenza) su un'immagine di sfondo background
# Inserisce la regione modificata nell'immagine originale
# Restituisce L'immagine di sfondo modificata con l'overlay sovrapposto.
def overlay_image_alpha(background, overlay, x, y):
    if overlay.shape[2] < 4:
        raise ValueError("L'overlay non ha il canale alfa!")

    scale_factor = 0.5
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))
    overlay = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError("Overlay fuori dai limiti dell'immagine di sfondo!")

    overlay_rgb = overlay[:, :, :3]
    alpha_mask = overlay[:, :, 3] / 255.0

    background_region = background[y:y + h, x:x + w]

    for c in range(0, 3):
        background_region[:, :, c] = (1 - alpha_mask) * background_region[:, :, c] + alpha_mask * overlay_rgb[:, :, c]

    background[y:y + h, x:x + w] = background_region

    # Aggiunta operazioni CPU-bound

    # Gaussian Blur (sfocatura)
    background = cv2.GaussianBlur(background, (7, 7), 0)

    # Sharpening Filter (aumento contrasto e nitidezza)
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    background = cv2.filter2D(background, -1, kernel_sharpening)

    # Transformation (Deformazioni)
    rows, cols, _ = background.shape
    src_pts = np.float32([[50, 50], [200, 50], [50, 200]])
    dst_pts = np.float32([[60, 60], [220, 50], [70, 210]])
    affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    background = cv2.warpAffine(background, affine_matrix, (cols, rows))

    # Laplacian Edge Detection (Evidenzia contorni)
    laplacian = cv2.Laplacian(background, cv2.CV_64F)
    background = cv2.convertScaleAbs(laplacian)

    # Histogram Equalization (miglioramento del contrasto)
    for i in range(3):
        background[:, :, i] = cv2.equalizeHist(background[:, :, i])

    return background


def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_dir, f"augmented_{i}.jpg"), img)

# Applica il processo di sovrapposizione e miglioramento dell'immagine
def process_image(img, overlay, x_offset, y_offset):
    return overlay_image_alpha(img.copy(), overlay, x_offset, y_offset)

# Esegue il processo in parallelo per più immagini
def process_images_parallel(images, overlay, x_offset, y_offset, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(delayed(process_image)(img, overlay, x_offset, y_offset) for img in images)
