import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os


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


#main

#import all images to list
img_dir_filepath = r"C:\Users\Jared B\Desktop\watermarked_imgs"
imgs = import_images(img_dir_filepath)

#storage vars where list index = image index
edges = []
imgs_processed = []
bounding_boxes = []
imgs_grad_horz = []
imgs_grad_vert = []
imgs_grad_mag = []

for img in imgs: 
    
    #process raw image
    img_processed = np.zeros(img.shape)
    for color_channel in range(img.shape[2]):
        img_processed[:,:,color_channel] = ndi.gaussian_filter(img[:,:,color_channel],sigma=1) #apply guassian filter to each color channel in the photo
                                                                
    img_processed = img_processed/255 #normlize img, 255 = max RGB numerical value for each color channel
    imgs_processed.append(img_processed)
    
    #extract edges
    edges.append(extract_image_edges(img_processed))        
    bounding_boxes.append(ndi.find_objects(edges[0])[0]) #find bounding box around objects where edges = True
    
    #find img gradients using sobel filter 
    imgs_grad_horz.append(ndi.sobel(img_processed, axis=0))
    imgs_grad_vert.append(ndi.sobel(img_processed, axis=1))

    mag2d_fn = np.vectorize(lambda x,y: (x**2 + y**2)** 0.5) #element wise magnitude fn
    img_grad_mag= np.zeros(img_processed.shape)
    
    for color_channel_n in range(img.shape[2]):
        # take gradient magnitude of vertical and horizontal directions (grad mag independent of color channel)
        img_grad_mag[:,:,color_channel_n] = mag2d_fn(imgs_grad_horz[-1][:,:,color_channel_n],imgs_grad_vert[-1][:,:,color_channel_n])  

    imgs_grad_mag.append(img_grad_mag)

#visualize
n = 0
c = 0
bounding_box_im = edges[n][bounding_boxes[n][0], bounding_boxes[n][1], c]
#plt.imshow(bounding_box_im)
#plt.show()