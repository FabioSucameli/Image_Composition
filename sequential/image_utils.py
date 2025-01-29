import os
import cv2
from glob import glob

# Caricamento di tutte le immagini
def load_images(input_dir, is_overlay=False):

    if is_overlay:
        overlay = cv2.imread(input_dir, cv2.IMREAD_UNCHANGED)
        return overlay
    else:
        image_paths = glob(os.path.join(input_dir, "*.png"))
        images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths]
        return images, image_paths

# Sovrappone un'immagine overlay (con canale alfa per la trasparenza) su un'immagine di sfondo background
# Inserisce la regione modificata nell'immagine originale
# Restituisce L'immagine di sfondo modificata con l'overlay sovrapposto.
def overlay_image_alpha(background, overlay, x, y):
    if overlay.shape[2] < 4:
        raise ValueError("L'overlay non ha il canale alfa!")

    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError("Overlay fuori dai limiti dell'immagine di sfondo!")

    overlay_rgb = overlay[:, :, :3]
    alpha_mask = overlay[:, :, 3] / 255.0  #

    background_region = background[y:y + h, x:x + w]

    for c in range(0, 3):
        background_region[:, :, c] = (1 - alpha_mask) * background_region[:, :, c] + alpha_mask * overlay_rgb[:, :, c]

    background[y:y + h, x:x + w] = background_region

    return background

# Salvataggio delle immagini modificate nella cartella di output
def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_dir, f"augmented_{i}.jpg"), img)
