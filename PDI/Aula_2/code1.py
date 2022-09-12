# Import threshold and gray convertor functions

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from PDI.src.pdi_utils import load_chess_image, show_image

"""
# load_image
tools_image = load_chess_image()

# Turn the image grayscale
gray_tools_image = rgb2gray(tools_image)

# Obtain the optimal thresh
thresh = threshold_otsu(gray_tools_image)

# Obtain the binary image by applying thresholding
binary_image = gray_tools_image > thresh


# Show the original image
show_image(tools_image, 'original image')
# Show the resulting binary image
show_image(binary_image, 'Binarized image')

"""