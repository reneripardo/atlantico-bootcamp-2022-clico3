import argparse
import os
import matplotlib.pyplot as plt

from ct_processing.ct_io import load_image



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_image', default=os.environ.get("PATH_IMAGE_FIRE"),
                    help='path image example')


    args = vars(ap.parse_args())

    image_fire = load_image(args['path_image'], 1)
    # plt.use('Agg')
    # plt.use('TkAgg')

    plt.subplot(131), plt.imshow(image_fire, cmap='rgb')
    plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(image_fire, cmap='gray')
    plt.title('xxx'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(image_fire, cmap='gray')
    plt.title('xxx'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()