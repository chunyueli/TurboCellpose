## AlignFOV2p
Align cross-day field-of-view (FOV) images based on Turboreg and Cellpose.
Cellpose is used to detect the ROIs masks of calcium-based FOV images. Turboreg is used to align ROIs masks across days.

## Notes
Cellpose package is written by Carsen Stringer and Marius Pachitariu. If you want to know more about Cellpose, please visit https://github.com/MouseLand/cellpose.
The python version Turboreg is from package pyStackReg (https://github.com/glichtner/pystackreg).

## System requirements
This code has been seriously tested on Windows 10. Codes requires Python 3.7.

## Installation
1. Install environment by running: conda env create -f [your file path to TuroCellpose/]environment.yaml
2. Enter environment by running: conda activate AlignFOV2p

## Usage
* cd [path to the upper level of the TuroCellpose]
* run python
* import AlignFOV2p
* Type: Tmatrices, regImages, regROIs=AlignFOV2p.AlignFOV2p("path to/examples/A5") or Tmatrices, regImages, regROIs=AlignFOV2p.AlignFOV2p("path to/examples/A6"), then the command will automatically register all the FOV images under the folder A5 or A6. Besides, the code will automatically extract ROI masks using the cellpose and save the masks under the folder of FOV images ("A5" or "A6").
* If the ROI masks have already been derived, providing a path to the ROI masks folder can save time. Run Tmatrices, regImages, regROIs=AlignFOV2p.AlignFOV2p("path_to_FOV_images", "path_to_ROI_masks")


## Outputs:
* Tmatrices: list of the transformation matrices
* regImages: list of the registered FOV images
* regROIs: list of the registered ROI masks
Besides, the Tmatrices and the registered FOV images as well as the registered ROIs masks will also be saved as a csv file and an image under the folder of FOV images.

## function parameters
* path_to_FOV: path of the folder containing FOV images
* path_to_masks: path of the folder containing ROIs mask. If the value is empty, the code will automatically extract ROI masks using the cellpose and save the masks under the folder of FOV images ("A5" or "A6"). If the ROI masks have already obtained, providing the path to the ROI masks folder can save time.
* templateID: choose which FOV image as a template for alignment.  Its default value is zero, indicating the first FOV image.
* transformation: str (default 'affine')
  - translation: X and Y translation
  - rigid_body: translation + rotation
  - scaled_rotation: translation + rotation + scaling
  - affine: translation + rotation + scaling + shearing
  - bilinear: non-linear transformation; does not preserve straight lines


## Dependencies
* pyStackReg
* pandas
* cellpose
* pystackreg
* numpy
* scikit-image
* matplotlib

## References
* Thevenaz, P., Ruttimann, U. E., & Unser, M. (1998). A pyramid approach to subpixel registration based on intensity. IEEE transactions on image processing, 7(1), 27-41.
* Stringer, C., Michaelos, M., & Pachitariu, M. (2020). Cellpose: a generalist algorithm for cellular segmentation. bioRxiv.
