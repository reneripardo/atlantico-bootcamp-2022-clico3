import argparse
import os
import matplotlib.pyplot as plt

from ct_processing.ct_image import (
    ContoursAreaSegmentation,
    equalize_Hist,
    gaussian_filter, 
    mean_filter, 
    median_filter,
    region_grow3D,
    threshold
)
from ct_processing.ct_io import load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_image', default=os.environ.get("PATH_IMAGE3"),
                    help='path image example')

    args = vars(ap.parse_args())

    image_input = load_image(args['path_image'], 0)
    image_seg = region_grow3D(image_input, [1], [10], 255, 255)

    plt.subplot(121), plt.imshow(image_input, cmap='gray')
    plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_seg, cmap='gray')
    plt.title('Segmentação'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()