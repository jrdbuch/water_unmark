import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct


def import_images(path_to_dir):
	filepaths = os.listdir(path_to_dir)
	images = []

	for filepath in filepaths:
		images.append(io.imread(path_to_dir + '\\' + filepath))

	return images 

def extract_image_edges(img, canny_sigma = 4):
    
    edges = np.zeros(img.shape,dtype=bool)
    
    for color_channel_n in range(img.shape[2]):
        edges[:,:,color_channel_n] = feature.canny(img[:,:,color_channel_n],sigma = 4)
        
    return edges

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
#main

#import all images to list
img_dir_filepath = r"C:\Users\Jared B\Desktop\watermarked_imgs"
imgs = import_images(img_dir_filepath)

#storage vars where list index = image index
imgs_edges = []
imgs_processed = []
bounding_boxes = []
imgs_Gx = []
imgs_Gy = []
imgs_Gmag = []
imgs_boundary = []

for img in imgs: 
    #process raw image    
    filter_img_fn = lambda n,img: ndi.gaussian_filter(img[:,:,n],sigma=1)
    img_processed = apply_fn_to_all_colorchannels(filter_img_fn,3,img) #run filter over img
    img_processed = img_processed/255 #normlize img, 255 = max RGB numerical value for each color channel
    
    #extract edges
    edge_detect_fn = lambda n,img :feature.canny(img[:,:,n],sigma = 4) 
    edges = apply_fn_to_all_colorchannels(edge_detect_fn,3,img)#run canny edge detection
    edges = np.array(edges,dtype=bool)
    bounding_box = ndi.find_objects(edges)[0] #find bounding box around objects where edges = True
    
    #find img gradients x,y, and magnitude using sobel filter 
    img_Gx = ndi.sobel(img_processed, axis=0)
    img_Gy = ndi.sobel(img_processed, axis=1)

    elementwise_2dmag_fn = lambda n,Gx,Gy: np.vectorize(lambda x,y: (x**2 + y**2)** 0.5)(Gx[:,:,n],Gy[:,:,n])  #element wise magnitude fn
    img_Gmag = apply_fn_to_all_colorchannels(elementwise_2dmag_fn, 3, img_Gx, img_Gy)
    
    #gather image boundary to be used for later poisson reconstruction
    img_boundary = np.zeros(img.shape) #new img with only the boundaries of the orginal img 
    img_boundary[0,:,:] = img[0,:,:]
    img_boundary[-1,:,:] = img[-1,:,:]
    img_boundary[:,0,:] = img[:,0,:]
    img_boundary[:,-1,:] = img[:,-1,:]
    
    #store single data sample in list 
    imgs_processed.append(img_processed)
    imgs_edges.append(edges)
    imgs_boundary.append(img_boundary)
    imgs_Gx.append(img_Gx)
    imgs_Gy.append(img_Gy)
    imgs_Gmag.append(img_Gmag)
    bounding_boxes.append(bounding_box)

#find pixelwise gradient medians over all images  
img_Gx_median = find_pixelwise_median(imgs_Gx)
img_Gy_median = find_pixelwise_median(imgs_Gy)

#find median img boundary
img_boundary_median = find_pixelwise_median(imgs_boundary) 

#crop to get bounding box of edgemap of Gmag

#poisson reconstruction
reconstruct_w_fn = lambda n,Gx,Gy,boundary: poisson_reconstruct(Gx[:,:,n],Gy[:,:,n],boundary[:,:,n])
w_estimate = apply_fn_to_all_colorchannels(reconstruct_w_fn, 3, np.multiply(img_Gx_median,edges),  np.multiply(img_Gy_median,edges), img_boundary_median)     
w_estimate = (w_estimate - np.min(w_estimate))/np.max(w_estimate)
#maybe need to only input Gx,Gy where edge = True, 0 everywhere else