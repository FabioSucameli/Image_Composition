import os
import cv2
import asyncio
from glob import glob

#Lettura di un'immagine in modo asincrono.
async def async_cv2_imread(path):
    return await asyncio.to_thread(cv2.imread, path, cv2.IMREAD_COLOR)

async def load_images(input_dir, max_images):
    image_paths = glob(os.path.join(input_dir, "*.png"))[:max_images]
    images = await asyncio.gather(*[async_cv2_imread(img_path) for img_path in image_paths])
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

# Esecuzione overlay_image_alpha in un thread separato per evitare il blocco del main event loop
async def process_image(img, overlay, x_offset, y_offset):
    return await asyncio.to_thread(overlay_image_alpha, img.copy(), overlay, x_offset, y_offset)

# Scrittura di un'immagine in modo asincrono
async def async_cv2_imwrite(path, img):
    await asyncio.to_thread(cv2.imwrite, path, img)

async def save_images(images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tasks = [async_cv2_imwrite(os.path.join(output_dir, f"augmented_{i}.png"), img) for i, img in enumerate(images)]
    await asyncio.gather(*tasks)
