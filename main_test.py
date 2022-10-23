import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import label
import cv2
from skimage.transform import resize
import os

BACKBONE = 'efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)


def preprocessing_HE(img_):
    
    hist, bins = np.histogram(img_.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img_2 = cdf[img_]
    
    return img_2  
        
def get_binary_mask (mask_, th_ = 0.5):
    mask_[mask_>th_]  = 1
    mask_[mask_<=th_] = 0
    return mask_
    
def ensemble_results (mask1_, mask2_, mask3_, mask4_, mask5_):
    
    mask1_ = get_binary_mask (mask1_)
    mask2_ = get_binary_mask (mask2_)
    mask3_ = get_binary_mask (mask3_)
    mask4_ = get_binary_mask (mask4_)
    mask5_ = get_binary_mask (mask5_)
    
    ensemble_mask = mask1_ + mask2_ + mask3_ + mask4_ + mask5_
    ensemble_mask[ensemble_mask<=2.0] = 0
    ensemble_mask[ensemble_mask> 2.0] = 1
    
    return ensemble_mask

def postprocessing_HoleFilling (mask_):
    
    ensemble_mask_post_temp = ndimage.binary_fill_holes(mask_).astype(int)
     
    return ensemble_mask_post_temp

def get_maximum_index (labeled_array):
    
    ind_nums = []
    for i in range (len(np.unique(labeled_array)) - 1):
        ind_nums.append ([0, i+1])
        
    for i in range (1, len(np.unique(labeled_array))):
        ind_nums[i-1][0] = len(np.where (labeled_array == np.unique(labeled_array)[i])[0])
        
    ind_nums = sorted(ind_nums)
    
    return ind_nums[len(ind_nums)-1][1], ind_nums[len(ind_nums)-2][1]
    
def postprocessing_EliminatingIsolation (ensemble_mask_post_temp):
        
    labeled_array, num_features = label(ensemble_mask_post_temp)
    
    ind_max1, ind_max2 = get_maximum_index (labeled_array)
    
    ensemble_mask_post_temp2 = np.zeros (ensemble_mask_post_temp.shape)
    ensemble_mask_post_temp2[labeled_array == ind_max1] = 1
    ensemble_mask_post_temp2[labeled_array == ind_max2] = 1    
    
    return ensemble_mask_post_temp2.astype(int)

def get_prediction(model_, img_org_):
    
    img_org_resize = cv2.resize(img_org_,(IMAGE_SIZE[0],IMAGE_SIZE[1]),cv2.INTER_AREA)
    img_org_resize_HE = preprocessing_HE (img_org_resize)    
    img_ready = preprocess_input(img_org_resize_HE)

    img_ready = np.expand_dims(img_ready, axis=0) 
    pr_mask = model_.predict(img_ready)
    pr_mask = np.squeeze(pr_mask)
    pr_mask = np.expand_dims(pr_mask, axis=-1)    
    return pr_mask[:,:,0]




path_image = os.environ.get("PATH_IMAGE")
path_base_model = os.environ.get("PATH_MODEL")

IMAGE_SIZE = (256,256,3)
model2 = sm.Unet(
    BACKBONE, 
    input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],IMAGE_SIZE[2]),
    classes=1, 
    activation='sigmoid',
    encoder_weights='imagenet'
)
model2.load_weights(path_base_model)

img = cv2.imread(path_image)
img_seg = get_prediction(model2, img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bottle_resized = resize(gray, img_seg.shape)

plt.subplot(121), plt.imshow(bottle_resized, cmap='gray')
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_seg, cmap='gray')
plt.title('seg'), plt.xticks([]), plt.yticks([])
plt.show()