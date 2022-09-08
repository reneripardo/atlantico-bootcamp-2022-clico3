import cv2


def load_image(path_image, flag_scale=0):
    """
    This method load an 2D image.

    Arguments:
        path_image {string} -- image path.
        flag_scale {int} -- 1 -> RGB and 0 -> GRAY.
    
    Returns:
        image {numpy} -- image.
    """

    image = cv2.imread(filename=path_image, flags=flag_scale)
    return image