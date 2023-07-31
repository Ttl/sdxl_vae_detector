Simple detector for identifying SDXL 1.0 generated images from VAE artifacts.

The simple original detector can be found in "v1" branch. The "master" branch
has improved detector that has over 99% accuracy in my test sets with positive
and negative examples.

To use the detector run: `./vae_detector_inference.py <input images>`. The
program supports glob syntax so to for example check all .png images in the
current folder use: `./vae_detector_inference.py "*.png"`.

The program prints image path and detection threshold from 0 to 1 on how likely
the image is decoded with SDXL 1.0 VAE with 1 being positive detection.

For more information see: https://hforsten.com/identifying-stable-diffusion-xl-10-images-from-vae-artifacts.html
