import asyncio
import time
import cv2
from image_utils import load_images, save_images, process_image

NUM_IMAGES = 500

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

    # Creazione di una lista di Task per elaborare le immagini in parallelo
    x_offset = 800
    y_offset = 500
    tasks = []
    for img in images:
        tasks.append(asyncio.create_task(process_image(img, overlay, x_offset, y_offset)))

    # Attendere il completamento di tutte le operazioni di sovrapposizione
    augmented_images = await asyncio.gather(*tasks)

    # Salvataggio asincrono
    await save_images(augmented_images, output_dir)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elaborazione completata in {elapsed_time:.2f} secondi.")

if __name__ == "__main__":
    asyncio.run(main())
