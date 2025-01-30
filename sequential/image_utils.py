import os
import cv2
from glob import glob

# Caricamento di tutte le immagini
def load_images(input_dir):
    image_paths = glob(os.path.join(input_dir, "*.png"))
    images = [cv2.imread(img_path, cv2.IMREAD_COLOR) for img_path in image_paths]
    #  print(f"Trovate {len(image_paths)} immagini")  # DEBUG
    return images, image_paths

# Sovrappone un'immagine overlay (ridotta del 50% con canale alfa per la trasparenza) su un'immagine di sfondo background
# Inserisce la regione modificata nell'immagine originale
# Restituisce L'immagine di sfondo modificata con l'overlay sovrapposto.
def overlay_image_alpha(background, overlay, x, y):
    if overlay.shape[2] < 4:
        raise ValueError("L'overlay non ha il canale alfa!")

    #print("Valore min/max canale alfa:", overlay[:, :, 3].min(), overlay[:, :, 3].max())  # DEBUG

    scale_factor = 0.5
    new_size = (int(overlay.shape[1] * scale_factor), int(overlay.shape[0] * scale_factor))
    overlay = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError("Overlay fuori dai limiti dell'immagine di sfondo!")

    #print(f"Sovrapposizione overlay su ({x}, {y}) con dimensioni ({w}, {h})")  # DEBUG

    overlay_rgb = overlay[:, :, :3]
    alpha_mask = overlay[:, :, 3] / 255.0

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
        #print(f"Salvata immagine: {path}")  # DEBUG
