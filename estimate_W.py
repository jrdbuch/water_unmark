import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct
import cv2


def import_images(path_to_dir):
	filepaths = os.listdir(path_to_dir)
	images = []

	for filepath in filepaths:
		images.append(io.imread(path_to_dir + '\\' + filepath))

	return images 

def apply_fn_to_all_colorchannels(fn,num_channels,*fn_args):
    #applies function "fn" to all channels independently with function inputs *fn_args
    #first input to custom function "fn" must be the channel number 
    
    for color_channel_n in range(num_channels):
        img_return_one_channel = fn(color_channel_n, *fn_args)
        
        if color_channel_n == 0:
            img_return = np.zeros(img_return_one_channel.shape + (num_channels,))
        
        img_return[:,:,color_channel_n] = img_return_one_channel
    
    return img_return

def find_pixelwise_median(img_list):
    img_median = np.zeros(img_list[0].shape)
    
    img_stack = np.asarray(img_list) #shape = (img_num,x,y,color_channel_n)
    
    for n in range(img_median.shape[2]):
        img_median[:,:,n] = np.median(img_stack[:,:,:,n],axis=0)
    
    return img_median

def resize_imgs(img_list,mode='centered_crop'):
    img_list_resized = []
    
    min_x = min([img.shape[0] for img in img_list])
    min_y = min([img.shape[1] for img in img_list])

    
#main

#import all images to list
img_dir_filepath = r"C:\Users\Jared B\Desktop\watermarked_imgs\123RF"
imgs = import_images(img_dir_filepath)

#storage vars where list index = image index
imgs_processed = []
imgs_Gx = []
imgs_Gy = []
imgs_Gmag = []
imgs_boundary = []
imgs_edges = []

for img in imgs: 
    #process raw image    
    filter_img_fn = lambda n,sigma,img: ndi.gaussian_filter(img[:,:,n],sigma=sigma)
    img_processed = apply_fn_to_all_colorchannels(filter_img_fn,3,1,img) #run filter over img
    
    #find img gradients x,y, and magnitude using sobel filter 
    img_Gx = ndi.sobel(img_processed, axis=0)
    img_Gy = ndi.sobel(img_processed, axis=1)

    elementwise_2Dmag_fn = lambda n,Gx,Gy: np.vectorize(lambda x,y: (x**2 + y**2)** 0.5)(Gx[:,:,n],Gy[:,:,n])  #element wise magnitude fn
    img_Gmag = apply_fn_to_all_colorchannels(elementwise_2Dmag_fn, 3, img_Gx, img_Gy)
    
    #find verbose edge map
    edge_detect_fn = lambda n,sigma,thresh_frac,img :feature.canny(img[:,:,n],sigma = sigma, high_threshold=thresh_frac*np.max(img)) 
    img_edges = apply_fn_to_all_colorchannels(edge_detect_fn,3,1,0.4,img_Gmag)
    
    #gather image boundary to be used for later poisson reconstruction
    img_boundary = np.zeros(img.shape) #new img with only the boundaries of the orginal img 
    img_boundary[0,:,:] = img[0,:,:]
    img_boundary[-1,:,:] = img[-1,:,:]
    img_boundary[:,0,:] = img[:,0,:]
    img_boundary[:,-1,:] = img[:,-1,:]
    
    #store single data sample in list 
    imgs_processed.append(img_processed)
    imgs_boundary.append(img_boundary)
    imgs_Gx.append(img_Gx)
    imgs_Gy.append(img_Gy)
    imgs_Gmag.append(img_Gmag)
    imgs_edges.append(img_edges)

for _ in range(1):
    
    ### Watermark estimation###
    
    #find pixelwise gradient medians over all images  
    img_Gx_median = find_pixelwise_median(imgs_Gx)
    img_Gy_median = find_pixelwise_median(imgs_Gy)
    img_Gmag_median = apply_fn_to_all_colorchannels(elementwise_2Dmag_fn, 3, img_Gx_median, img_Gy_median)
    
    #find median img boundary
    img_boundary_median = find_pixelwise_median(imgs_boundary) 
    
    #extract edges
    img_edges_median = apply_fn_to_all_colorchannels(edge_detect_fn,3,1,0.3,img_Gmag_median).astype(bool)#run canny edge detection
    bounding_box = ndi.find_objects(img_edges_median)[0] #find bounding box around objects where edges = True
    
    #use edges to locate the gradients of interest for poisson reconstruction 
    img_Gx_reconstruct = np.multiply(img_Gx_median,img_edges_median)
    img_Gy_reconstruct = np.multiply(img_Gy_median,img_edges_median)
    
    ###Watermark detection### not needed if bounding box for the template is the entire image 
    #match = cv2.matchTemplate(img_edges_median[:,:,0].astype(np.uint8),imgs_edges[0][:,:,0].astype(np.uint8),method=eval('cv2.TM_CCOEFF')) #template, img
        
#poisson reconstruction
reconstruct_w_fn = lambda n,Gx,Gy,boundary: poisson_reconstruct(Gy[:,:,n],Gx[:,:,n],boundary[:,:,n])
w_estimate = apply_fn_to_all_colorchannels(reconstruct_w_fn, 3, img_Gx_reconstruct, img_Gy_reconstruct, img_boundary_median)    