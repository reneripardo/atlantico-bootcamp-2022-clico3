import argparse
import os
import matplotlib.pyplot as plt

from ct_processing.ct_image import (
    region_grow3D,
    threshold
)
from ct_processing.ct_io import load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_image', default=os.environ.get("IMG_LIDC_IDRI_01111"),
                    help='path image example')

    args = vars(ap.parse_args())

    image_input = load_image(args['path_image'], 0)
    _, image_otsu = threshold(image_input)

    y_ct = int(image_input.shape[0]/2)
    x_ct = int(image_input.shape[1]/2)    
    
    image_cr = region_grow3D(
        image_otsu, 
        [x_ct-1, x_ct-2, x_ct, x_ct+1, x_ct+2], 
        [y_ct-1, y_ct-2, y_ct, y_ct+1, y_ct+2], 
        255, 
        255
    )

    plt.subplot(131), plt.imshow(image_input, cmap='gray')
    plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(image_otsu, cmap='gray')
    plt.title('threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(image_cr, cmap='gray')
    plt.title('Seg Crescimento de regioes'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()