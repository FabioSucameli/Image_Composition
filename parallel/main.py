import time
import asyncio
import cv2
from image_utils import overlay_image_alpha, load_images, save_images

NUM_IMAGES = 100

async def main():
    input_dir = "../input_img"
    output_dir = "../output_img"
    overlay_path = "../cat.png"

    start_time = time.time()

    # Caricamento immagini asincrono e overlay
    images, image_paths = await load_images(input_dir, NUM_IMAGES)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    if overlay is None:
        raise FileNotFoundError("Immagine del gatto non trovata!")

    # Applicazione della sovrapposizione
    augmented_images = []
    for img in images:
        x_offset = 800
        y_offset = 500
        augmented_img = overlay_image_alpha(img.copy(), overlay, x_offset, y_offset)
        augmented_images.append(augmented_img)

    # Salvataggio asincrono
    await save_images(augmented_images, output_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elaborazione completata in {elapsed_time:.2f} secondi.")

if __name__ == "__main__":
    asyncio.run(main())