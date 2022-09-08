import cv2
import numpy as np

def ContoursAreaSegmentation(image_binary):
    contours, hierarchy = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    all_areas=[]
    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    # indice do contorno de maior area interna 
    ind_max_area = all_areas.index(max(all_areas)) 
    img_zeros = np.zeros_like(image_binary)
    # Retorna imagem com o contorno que possui maior area interna  
    image_segmentated = cv2.drawContours(img_zeros, [contours[ind_max_area]], 0, 255, -1)
    image_segmentated[image_segmentated!=0] = 1

    return image_segmentated

def threshold(image_in):
    """Apply threshold.
        Arguments:
            image_in {numpy.array} -- image input.
        Returns:
            threshold_value {int} -- threshold image.
            threshold_image {numpy.array} -- threshold value.
    """
    tech = cv2.THRESH_BINARY+cv2.THRESH_OTSU
    threshold_value, threshold_image = cv2.threshold(image_in, 0, 255, tech)
    return threshold_value, threshold_image

def equalize_Hist(image_in):
    """Apply equalize histogram.
        Arguments:
            image_in {numpy.array} -- image input.
        Returns:
            image_out {numpy.array} -- image ouput.
    """
    image_out = cv2.equalizeHist(image_in)
    return image_out

def mean_filter(image_in, kernel=3):
    """Apply mean filter.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- image ouput.
    """
    image_out = cv2.blur(image_in, (kernel,kernel))
    return image_out

def median_filter(image_in, kernel=3):
    """Apply median filter.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- image ouput.
    """
    image_out = cv2.medianBlur(image_in, kernel)
    return image_out

def gaussian_filter(image_in, kernel=3):
    """Apply gaussian filter.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- image ouput.
    """
    image_out = cv2.GaussianBlur(image_in, (kernel,kernel), 0)
    return image_out

def laplacian_filter(image_in):
    """Apply laplacian filter.
        Arguments:
            image_in {numpy.array} -- image input.
        Returns:
            image_out {numpy.array} -- image ouput.
    """
    image_out = cv2.Laplacian(image_in, cv2.CV_64F)
    return image_out

def sobel_filter(image_in, kernel=3):
    """Apply sobel filter.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_outx {numpy.array} -- x-axis output image.
            image_outy {numpy.array} -- y-axis output image.
    """
    image_outx = cv2.Sobel(image_in, cv2.CV_64F, 1, 0, ksize=kernel)
    image_outy = cv2.Sobel(image_in, cv2.CV_64F, 0, 1, ksize=kernel)
    return image_outx, image_outy

def morpho_erode(image_in, kernel=3):
    """Apply erosion.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- output image.
    """
    image_out = cv2.erode(image_in, np.ones((kernel,kernel),np.uint8), iterations=1)
    return image_out

def morpho_dilate(image_in, kernel=3):
    """Apply delation.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- output image.
    """
    image_out = cv2.dilate(image_in, np.ones((kernel,kernel),np.uint8), iterations=1)
    return image_out

def morpho_open(image_in, kernel=3):
    """Erosion followed by dilation.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- output image.
    """
    image_out = cv2.morphologyEx(image_in, cv2.MORPH_OPEN, np.ones((kernel,kernel),np.uint8))
    return image_out

def morpho_close(image_in, kernel=3):
    """Dilation followed by Erosion.
        Arguments:
            image_in {numpy.array} -- image input.
            kernel {int} -- neighborhood size.
        Returns:
            image_out {numpy.array} -- output image.
    """
    image_out = cv2.morphologyEx(image_in, cv2.MORPH_CLOSE, np.ones((kernel,kernel),np.uint8))
    return image_out