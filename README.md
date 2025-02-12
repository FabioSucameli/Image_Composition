# Image Composition

**Image composition** is a technique that combines multiple images by overlaying them using an alpha channel (transparency). It is widely used in **image augmentation** for training **Convolutional Neural Networks (CNNs)** in object detection tasks.

By using **parallel alpha blending**, it speeds up processing while making datasets more diverse and models more robust. This method allows the same object to be applied across multiple images or different positions within a single image, creating more varied and realistic training data.

## Applications
- **Computer Vision**: Enhancing training datasets for object detection and classification.
- **Image Augmentation**: Creating synthetic training data by blending foreground objects with various backgrounds.
- **Parallel Processing**: Efficiently applying transformations to multiple images simultaneously.

---
For my project, I performed **image composition** by overlaying cat images onto background images of streets from the **Cityscapes dataset**: [https://www.cityscapes-dataset.com/dataset-overview/](https://www.cityscapes-dataset.com/dataset-overview/). This process involved using an **alpha channel** to blend the foreground (cat) with the background (urban streets).  

Below there is an example of how the image composition process works:
![Image Composition](image_composition.gif)
 
The goal of this technique was to create a **diverse and realistic dataset** by placing the same object (cats) in different environments. By using **parallel alpha blending**, I was able to efficiently generate multiple variations of training images, making the dataset more diverse. This approach is particularly useful for **image augmentation**, as it helps deep learning models generalize better by exposing them to varied contexts.
