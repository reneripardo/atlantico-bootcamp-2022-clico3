import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import convex_hull_image

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

def region_grow3D(array_imgCR, point_x, point_y, tom_max, tom_min):
    """
    Application of the regional growth technique.
          Arguments:
              array_img {numpy.array} -- original image.
              point_y {list} -- lista com as coordenadas das linhas.
              point_x {list} -- lista com as coordenadas das colunas.
              tom_max {int}  -- tom de cinza maior ou igual da semente.
              tom_min {int}  -- tom de cinza menor ou igual da semente.
          Returns:
              array_img_seg {numpy.array} -- segemntação.
    """

    #zeros image for 3D targeting
    array_img_seg = np.zeros(array_imgCR.shape).astype(np.uint8)

    #kernel for one-pixel neighborhood
    array8u = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=int)

    #set image coordinates with non-zero gray (slice, line and column)
    row = []
    col = []

    for len_array_seed in range(0, len(point_y)):
        row.append(point_y[len_array_seed])
        col.append(point_x[len_array_seed])
        array_img_seg[row[len_array_seed], col[len_array_seed]] = 1

    cont = 0
    while cont < len(row):
        if np.abs(array_imgCR.size-len(point_y)) == cont:
            break
        for i in range(-1, 2):
            if (row[cont] + i > 0) and (row[cont] + i < (array_img_seg.shape[0]-1)):
                for j in range(-1, 2):
                    if (col[cont]+j > 0) and (col[cont]+j < (array_img_seg.shape[1]-1)):  # nao ir nas bordas
                        if (array_imgCR[row[cont] + i, col[cont] + j] <= tom_max) and \
                           (array_imgCR[row[cont] + i, col[cont] + j] >= tom_min) and \
                           (array_img_seg[row[cont] + i, col[cont] + j] == 0) and \
                           (array8u[1 + i, 1 + j] == 1):
                           
                           array_img_seg[row[cont] + i, col[cont] + j] = 1
                           row.append(row[cont] + i)
                           col.append(col[cont] + j)

        cont += 1
    return array_img_seg

def nodule_location(image_bin_nodule):
    """Nodule location due to the presence of pleura.
        Arguments:
            image_bin_nodule {numpy.array} -- 2D binary (0 or 1) image of the nodule.
        Returns:
            value {int} -- 0 -> non-juxtaural nodule,
                           1 -> juxtapleural nodule.
    """
    image_borders = np.ones_like(image_bin_nodule)
    image_borders[1:-1,1:-1,1:-1] = 0
    area_border = np.count_nonzero(image_borders)
    del image_borders

    ind_non_zeros = np.where(image_bin_nodule != 0)
    cont_area = 0
    shape_y = image_bin_nodule.shape[0]
    shape_x = image_bin_nodule.shape[1]
    for i in range(len(ind_non_zeros[0])):
        z = ind_non_zeros[0][i]
        y = ind_non_zeros[1][i]
        x = ind_non_zeros[2][i]

        if y==0 or y==shape_y-1 or x==0 or x == shape_x-1:
            cont_area+=1

    del ind_non_zeros
    if cont_area >= 0.1*area_border:
        return 1
    return 0

def get_convex_hull_image(array_image):
    """apply convex hull in image.
        Arguments:
            array_image {numpy.array} -- input 3D image.
        Returns:
            array_image_convex_hull {numpy.array} -- 3D image.
    """
    array_image_convex_hull = np.zeros(array_image.shape, 'float64')
    array_image_convex_hull = convex_hull_image(array_image)

    return array_image_convex_hull

def component_largest_connected(array_image):
    """get component largest connected.
        Arguments:
            array_image {numpy.array} -- input image.
        Returns:
            image_component_largest_connected {numpy.array} -- array image binary (component largest connected).
    """
    array_image[array_image != 0] = 1
    labels = label(array_image)
    image_component_largest_connected = labels == np.argmax(np.bincount(labels.flat, weights=array_image.flat))
    image_component_largest_connected.dtype = np.uint8

    return image_component_largest_connected

def get_pleura_nodule(array_image):
    """ get pleura nodule.
        Arguments:
            array_image {numpy.array} -- input image binary.
        Returns:
            array_image_pleura {numpy.array} -- thresholded image of the pleura.
    """
    array_nzero = np.array(array_image)

    array_complement_n_zero = array_nzero.max() - array_image

    if array_complement_n_zero.max() != 0:
        array_negative_nzero_largest_component = component_largest_connected(array_complement_n_zero)

        array_n_two = np.array(array_nzero*get_convex_hull_image(array_negative_nzero_largest_component))

        array_image_pleura = np.array(array_n_two.max() - array_n_two)*array_image

        return array_image_pleura
    else:
        return np.zeros(array_image.shape, np.uint8)

def pipleline(array_image):
    pleura = nodule_location(array_image)
    if pleura is True:
          return get_pleura_nodule(array_image)
    return array_image

