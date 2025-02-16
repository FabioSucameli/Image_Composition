# I/O-Bound Approach

This project implements an **I/O-bound approach** to image composition, where an overlay image (a cat) is placed onto background images of streets asynchronously. Instead of using **CPU-bound parallelization**, I opted for an **I/O-bound strategy** because the main bottleneck was not the computational cost of overlaying images, but rather the **loading and saving of image files**.
