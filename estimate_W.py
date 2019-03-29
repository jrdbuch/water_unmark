import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct
import cv2

##TO DO make agnostic to shape


class WaterUnmark:
    def __init__(self):
        self._imgs_raw = []
        self._imgs_processed = []
        self._imgs_Gx = []
        self._imgs_Gy = []
        self._imgs_Gmag = []
        self._imgs_boundary = []
        self._imgs_edges = []

        #replace this with numpy func
        self._elementwise_2Dmag_fn = lambda n, Gx, Gy: np.vectorize(lambda x, y:
                                    (x ** 2 + y ** 2) ** 0.5)(Gx[:, :, n], Gy[:, :,n])  # element wise magnitude fn

        self._edge_detect_fn = lambda n, sigma, thresh_frac, img: feature.canny(img[:, :, n], sigma=sigma,
                                                                          high_threshold=thresh_frac * np.max(img))

    def import_training_images(self, path_to_dir):
        img_paths = os.listdir(path_to_dir)
        self._imgs_raw = []

        for img_path in img_paths:
            self._imgs_raw.append(io.imread(os.path.join(path_to_dir,img_path)))

    @staticmethod
    def _apply_fn_to_all_colorchannels(fn, num_channels, *fn_args):
        # applies function "fn" to all channels independently with function inputs *fn_args
        # first input to custom function "fn" must be the channel number
        # REFACTOR THIS

        for color_channel_n in range(num_channels):
            img_return_one_channel = fn(color_channel_n, *fn_args)

            if color_channel_n == 0:
                img_processed = np.zeros(img_return_one_channel.shape + (num_channels,))

            img_processed[:,:,color_channel_n] = img_return_one_channel

        return img_processed

    @staticmethod
    def _find_pixelwise_median(img_list):
        # finds the pixelwise median over a collection of imgs
        # TO DO: Replace with numpy function

        img_median = np.zeros(img_list[0].shape)

        img_stack = np.asarray(img_list) #shape = (img_num,x,y,color_channel_n)

        for n in range(img_median.shape[2]):
            img_median[:,:,n] = np.median(img_stack[:,:,:,n],axis=0)

        return img_median

    def _extract_img_boundaries(sel, img):
        # TO DO: make this robust for shifted watermarks
        # gather image boundary to be used for later poisson reconstruction
        img_boundary = np.zeros(img.shape)  # new img with only the boundaries of the orginal img
        img_boundary[0, :, :] = img[0, :, :]
        img_boundary[-1, :, :] = img[-1, :, :]
        img_boundary[:, 0, :] = img[:, 0, :]
        img_boundary[:, -1, :] = img[:, -1, :]

        return img_boundary

    def _preprocess_img (self, img):
        # preprocess raw image
        filter_img_fn = lambda n, sigma, img: ndi.gaussian_filter(img[:, :, n], sigma=sigma)
        img_processed = self._apply_fn_to_all_colorchannels(filter_img_fn, 3, 1, img)  # run filter over img

        return img_processed


    def _extract_img_gradients(self, img):
        # find img gradients x,y, and magnitude using sobel filter
        img_Gx = ndi.sobel(img, axis=0)
        img_Gy = ndi.sobel(img, axis=1)

        img_Gmag = self._apply_fn_to_all_colorchannels(self._elementwise_2Dmag_fn, 3, img_Gx, img_Gy)

        return img_Gx, img_Gy, img_Gmag

    def _extract_edgemap(self, img_Gmag):
        # find verbose edge map

        img_edges = self._apply_fn_to_all_colorchannels(self._edge_detect_fn, 3, 1, 0.4, img_Gmag)

        return img_edges

    def reconstruct_watermark(self, img_Gx, img_Gy, img_boundary):
        reconstruct_w_fn = lambda n, Gx, Gy, boundary: poisson_reconstruct(Gy[:, :, n], Gx[:, :, n], boundary[:, :, n])
        w_estimate = self._apply_fn_to_all_colorchannels(reconstruct_w_fn, 3, img_Gx, img_Gy, img_boundary)

        return w_estimate

    def _find_elementwise_(self):
        pass

    def estimate_watermark(self):

        #iterate over raw training imgs
        for img in self._imgs_raw:

            img_processed = self._preprocess_img(img)
            img_Gx, img_Gy, img_Gmag =  self._extract_img_gradients(img_processed)
            img_edges = self._extract_edgemap(img)
            img_boundary = self._extract_img_boundaries(img)

            # store single data sample in list
            self._imgs_processed.append(img_processed)
            self._imgs_boundary.append(img_boundary)
            self._imgs_Gx.append(img_Gx)
            self._imgs_Gy.append(img_Gy)
            self._imgs_Gmag.append(img_Gmag)
            self._imgs_edges.append(img_edges)

        ### Watermark estimation###

        # find pixelwise gradient medians over all images
        self._imgs_Gx_median = self._find_pixelwise_median(self._imgs_Gx)
        self._imgs_Gy_median = self._find_pixelwise_median(self._imgs_Gy)
        self._imgs_Gmag_median = self._apply_fn_to_all_colorchannels(self._elementwise_2Dmag_fn, 3, self._imgs_Gx_median, self._imgs_Gy_median)

        # find median img boundary
        self._imgs_boundary_median = self._find_pixelwise_median(self._imgs_boundary)

        # extract edges
        self._imgs_edges_median = self._apply_fn_to_all_colorchannels(self._edge_detect_fn, 3, 1, 0.3, self._imgs_Gmag_median).astype(bool)  # run canny edge detection
        #bounding_box = ndi.find_objects(img_edges_median)[0]  # find bounding box around objects where edges = True

        # use edges to locate the gradients of interest for poisson reconstruction
        img_Gx_reconstruct = np.multiply(self._imgs_Gx_median, self._imgs_edges_median)
        img_Gy_reconstruct = np.multiply(self._imgs_Gy_median, self._imgs_edges_median)

        print(self._imgs_boundary_median.shape)

        plt.imshow(poisson_reconstruct(img_Gy_reconstruct[:,:,2], img_Gx_reconstruct[:,:,2], self._imgs_boundary_median[:,:,2]))
        plt.show()

        ###Watermark detection### not needed if bounding box for the template is the entire image
        #match = cv2.matchTemplate(img_edges_median[:,:,0].astype(np.uint8),imgs_edges[0][:,:,0].astype(np.uint8),method=eval('cv2.TM_CCOEFF')) #template, img

        # poisson reconstruction
        w_estimate = self.reconstruct_watermark(img_Gx_reconstruct, img_Gy_reconstruct, self._imgs_boundary_median)



def resize_imgs(img_list,mode='centered_crop'):
    img_list_resized = []
    
    min_x = min([img.shape[0] for img in img_list])
    min_y = min([img.shape[1] for img in img_list])


if __name__ == "__main__":

    wu = WaterUnmark()
    wu.import_training_images(u'/home/jaredbuchanan/water_unmark/examples/123RF')
    w = wu.estimate_watermark()

    print(type(w))



