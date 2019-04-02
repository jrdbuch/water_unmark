import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct
import cv2

##TO DO make agnostic to shape

#replace this with numpy func



def import_training_images(path_to_dir):
    #imgs imported with 3 colors channels of ints ranging from 0-255
    img_paths = os.listdir(path_to_dir)
    imgs_raw = []

    for img_path in img_paths:
        imgs_raw.append(mpimg.imread(os.path.join(path_to_dir,img_path)))

    return imgs_raw

def extract_img_boundaries(img):
    # TO DO: make this robust for shifted watermarks
    # gather image boundary to be used for later poisson reconstruction
    img_boundary = np.zeros(img.shape)  # new img with only the boundaries of the orginal img
    img_boundary[0, :, :] = img[0, :, :]
    img_boundary[-1, :, :] = img[-1, :, :]
    img_boundary[:, 0, :] = img[:, 0, :]
    img_boundary[:, -1, :] = img[:, -1, :]

    return img_boundary

def preprocess_img(img):
    # preprocess raw image

    img_processed = np.zeros(img.shape)

    for n in range(img.shape[-1]):
        img_processed[:,:,n] = ndi.gaussian_filter(img[:, :, n], sigma=1)

    return img_processed


def extract_img_gradients(img):
    # find img gradients x,y, and magnitude using sobel filter
    img_Gx = ndi.sobel(img, axis=0)/8
    img_Gy = ndi.sobel(img, axis=1)/8

    img_Gmag = np.linalg.norm(np.stack([img_Gx, img_Gy], axis=-1), axis= -1)

    return img_Gx, img_Gy, img_Gmag

def extract_edgemap(img_Gmag):
    # find verbose edge map

    edge_detect_fn = lambda n, sigma, thresh_frac, img: feature.canny(img[:, :, n], sigma=sigma,
                                                                      high_threshold=thresh_frac * np.max(img))

    img_edges = np.zeros(img_Gmag.shape)

    for n in range(img_Gmag.shape[-1]):
        img_edges[:,:,n] = edge_detect_fn(n, 1, 0.4, img_Gmag)

    return img_edges

def reconstruct_watermark(img_Gx, img_Gy, img_boundary):

    w_estimate = np.zeros(img_Gx.shape)

    for n in range(img_Gx.shape[-1]): #iterate over color channels and perform poisson reconstruction
        w_estimate[:,:,n] = poisson_reconstruct(img_Gx[:,:,n], img_Gy[:,:,n], img_boundary[:,:,n])

    return w_estimate

def normalize(n, img):
    #normalize so all values are between 0 and 1
    img_normalize = img[:,:,n] / np.amax(img[:,:,n])

    return img_normalize

def detect_watermark():
    # extract edges
    #imgs_edges_median = apply_fn_to_all_colorchannels(edge_detect_fn, 3, 1, 0.3, imgs_Gmag_median).astype(bool)  # run canny edge detection

    #find bounding box given the edges
    # bounding_box = ndi.find_objects(img_edges_median)[0]  # find bounding box around objects where edges = True

    ###Watermark detection### not needed if bounding box for the template is the entire image
    # match = cv2.matchTemplate(img_edges_median[:,:,0].astype(np.uint8),imgs_edges[0][:,:,0].astype(np.uint8),method=eval('cv2.TM_CCOEFF')) #template, img

    pass

def threshold_img(img, threshold):
    img[np.abs(img) < threshold] = 0

    return img


def estimate_watermark(imgs_raw):

    imgs_processed = []
    imgs_Gx = []
    imgs_Gy = []
    imgs_Gmag = []
    imgs_boundary = []
    imgs_edges = []

    #iterate over raw training imgs
    for img in imgs_raw:

        img_processed = preprocess_img(img)
        img_Gx, img_Gy, img_Gmag =  extract_img_gradients(img_processed)
        img_edges = extract_edgemap(img)
        img_boundary = extract_img_boundaries(img)

        # store single data sample in list
        imgs_processed.append(img_processed)
        imgs_boundary.append(img_boundary)
        imgs_Gx.append(img_Gx)
        imgs_Gy.append(img_Gy)
        imgs_Gmag.append(img_Gmag)
        imgs_edges.append(img_edges)

    ### Watermark detection###
    #make sure watermarks are same img size after detection in img
    #also might need to do some image resizing
    #can skip for now

    # find pixelwise gradient medians over all images
    imgs_Gx_median = np.median(np.stack(imgs_Gx, axis=-1), axis=-1)
    imgs_Gy_median = np.median(np.stack(imgs_Gy, axis=-1), axis=-1)
    imgs_Gy_median = threshold_img(imgs_Gy_median, 3)
    imgs_Gx_median = threshold_img(imgs_Gx_median, 3)
    #imgs_Gmag_median = apply_fn_to_all_colorchannels(elementwise_2Dmag_fn, 3, imgs_Gx_median, imgs_Gy_median)
    imgs_Gmag_median = np.linalg.norm(np.stack([imgs_Gx_median, imgs_Gy_median], axis=-1), axis=-1)

    # poisson reconstruction
    print('Reconstructing Watermark')
    w_estimate = reconstruct_watermark(imgs_Gx_median, imgs_Gy_median, np.zeros(imgs_Gx_median.shape))


    #########debug
    print('imgs_Gx_median ', np.amax(imgs_Gx_median[:,:,0]), np.amin(imgs_Gx_median[:,:,0]), np.mean(imgs_Gx_median[:,:,0]))
    print(np.amax(w_estimate))

    plt.figure(1)
    plt.imshow(w_estimate.astype(int))
    plt.figure(2)
    plt.imshow(imgs_Gx_median[:,:,0].astype(int))
    plt.figure(3)
    plt.imshow(imgs_Gy_median.astype(int))
    plt.figure(4)
    plt.imshow(imgs_Gmag_median.astype(int))

    return w_estimate


def detect_watermark():
    pass


