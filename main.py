import argparse
import os
import matplotlib.pyplot as plt

from ct_processing.ct_image import (
    ContoursAreaSegmentation,
    equalize_Hist,
    gaussian_filter, 
    mean_filter, 
    median_filter,
    threshold
)
from ct_processing.ct_io import load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_image', default=os.environ.get("PATH_IMAGE"),
                    help='path image example')
    ap.add_argument('-i2', '--path_image2', default=os.environ.get("PATH_IMAGE2"),
                    help='path image example')


    args = vars(ap.parse_args())

    image_input = load_image(args['path_image'], 0)
    image_input2 = load_image(args['path_image2'], 0)

    #--------> pre processing
    image_mean_filter = mean_filter(image_input, 5)
    image_median_filter = median_filter(image_input, 5)
    image_gaussian_filter = gaussian_filter(image_input, 5)


    # Equalization
    image_equ = equalize_Hist(image_input2)

    # Otsu's thresholding
    _, image_input_otsu = threshold(image_input2)
    _, image_equ_otsu = threshold(image_equ)

    plt.subplot(221), plt.imshow(image_input, cmap='gray')
    plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(image_mean_filter, cmap='gray')
    plt.title('Filtro da média (kernel=5x5)'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(image_median_filter, cmap='gray')
    plt.title('Filtro da mediana (kernel=5x5)'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(image_gaussian_filter, cmap='gray')
    plt.title('Filtro gaussiana (kernel=5x5)'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(221), plt.imshow(image_input2, cmap='gray')
    plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(image_equ, cmap='gray')
    plt.title('Com equalização'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(image_input_otsu, cmap='gray')
    plt.title('Limiarização com Otsu \n na imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(image_equ_otsu, cmap='gray')
    plt.title('Limiarização com Otsu \n na imagem com equalização'), plt.xticks([]), plt.yticks([])
    plt.show()


    #--------> segemtnation
    image_seg = ContoursAreaSegmentation(image_input_otsu)

    plt.subplot(121), plt.imshow(image_input2, cmap='gray')
    plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(image_input_otsu, cmap='gray')
    plt.title('xx'), plt.xticks([]), plt.yticks([])
    plt.subplot(123), plt.imshow(image_input_otsu, cmap='gray')
    plt.title('xx'), plt.xticks([]), plt.yticks([])
    plt.show()




if __name__ == "__main__":
    main()