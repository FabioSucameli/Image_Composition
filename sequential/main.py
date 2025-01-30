import os
import cv2
from image_utils import load_images, overlay_image_alpha, save_images

def main():
    input_dir = "../input_img"  # Directory di input (immagini delle strade di CityScapes)
    output_dir = "../output_img"  # Directory di output
    overlay_path = "../cat.png"  # Oggetto da sovrapporre (gatto nero)

    # Caricamento delle immagini e overlay
    images, image_paths = load_images(input_dir)
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

    # Salvataggio delle immagini modificate nella relativa cartella di output
    save_images(augmented_images, output_dir)

if __name__ == "__main__":
    main()
