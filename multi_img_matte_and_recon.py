import numpy as np
import copy
from estimate_W import extract_img_gradients
import scipy
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct
import cv2


def Edata(i,w,a,j):
    """
    Cost associated with the deviation of W applied to I via the formation model (Jform) as compared with the actual J

    :param i: Natural Image, single pixel
    :param w: Watermark image, single pixel
    :param a: alpha matte, single pixel
    :param j: natural image with watermark, single pixel
    :return:
    """

    jform = add_watermark(w, i, a)
    cost = L1_approx(np.square(np.absolute(jform - j)))
            
    return cost
    
def Ereg_ig(igx, igy , agx, agy):
    """
    :param igx:
    :param igy:
    :param agx:
    :param agy:
    :return:
    """

    return L1_approx(np.square(np.multiply(np.absolute(agx), igx)) +
                     np.square(np.multiply(np.absolute(agy), igy)))

def Ereg_ag(agx,agy):
    return L1_approx(np.square(agx) + np.square(agy))
    
def Ef(wkgx, wkgy, w_init_gx, w_init_gy, A):
    """
    :param wkg: single pixel gradient for current watermark estimate for image k
    :param w_init_g: single pixel gradient of initial watermark estimate
    :param a: single pixel alpha
    :return:
    """

    wkg = np.stack([wkgx, wkgy], axis=-1)  # [x, y, n, gx/gy]
    w_init_g = np.stack([w_init_gx, w_init_gy], axis=-1) # [x, y, n, gx/gy]

    return L1_approx(np.multiply(np.linalg.norm(wkg - w_init_g, ord=2, axis=-1), A))
    
def Eaux(w,wk):
    return np.absolute(w - wk)

def L1_approx(s2,e=0.001):
    
    return np.sqrt(s2+e**2)

def add_watermark(W, I, A):
    return np.multiply(W, A) + np.multiply(I, 1-A)


def cost(ik, wk, a, jk, agx, agy, w_init_gx, w_init_gy, w):

    ikgx, ikgy, _ = extract_img_gradients(ik)
    wkgx, wkgy, _ = extract_img_gradients(wk)

    cost = Edata(ik, wk, a, jk) + Ereg_ig(ikgx, ikgy, agx, agy) + \
           Ereg_ag(agx, agy) + Ef(wkgx, wkgy, w_init_gx, w_init_gy, a) + Eaux(w, wk)

    return np.sum(cost)


def image_watermark_decomposition(W, W_init, A, Jk):
    """
    :param W: current estimate of global estimate of watermark
    :param W_init:
    :param A: current estimate of matte matrix
    :param Jk: single training image with watermark
    :return:
    """

    Agx, Agy, _ = extract_img_gradients(A)
    W_init_gx, W_init_gy, _ = extract_img_gradients(W_init)

    print(np.stack([Jk, W], axis=-1).shape)


    #get rid of this with args
    cost_fn = lambda var: cost(var[:,:,:,0],
                               var[:,:,:,1],
                               np.dstack([A]*3), #add color channel dim
                               Jk,
                               np.dstack([Agx]*3),
                               np.dstack([Agy]*3),
                               W_init_gx, W_init_gy, W)

    print(cost_fn(np.stack([Jk, W], axis=-1)))

    results = scipy.optimize.minimize(fun=cost_fn,
                                      x0=np.stack([Jk, W], axis=-1),
                                      options={"maxiter": 2}, jac=False)

    Ik = results.x[:,:,:,0]
    Wk = results.x[:,:,:,1]

    """
    #iterate over individual pixels
    for x in range(Jk.shape[0]):
        for y in range(Jk.shape[1]):
            for n in range(Jk.shape[2]):
                #calculate cost for single pixel in image
                print(x,y,n)



                #optimize Ik, Wk terms for single pixel
                results = scipy.optimize.minimize(cost_fn, np.array([Ik[x,y,n], Wk[x,y,n]]), options={"maxiter":2}, jac=False)

                Ik[x,y,n] = results.x[0]
                Wk[x,y,n] = results.x[1]
                
    """


    return Ik, Wk

def matte_update():
    for x in range(Jk.shape[0]):
        for y in range(Jk.shape[1]):
            for n in range(Jk.shape[3]):
                pass


