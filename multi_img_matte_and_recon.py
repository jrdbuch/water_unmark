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
    cost = L1_approx(np.absolute(jform - j)**2)
            
    return cost
    
def Ereg_ig(igx, igy , agx, agy):
    """
    :param igx:
    :param igy:
    :param agx:
    :param agy:
    :return:
    """

    return L1_approx(np.absolute(agx)*igx**2 + np.absolute(agy)*igy**2)

def Ereg_ag(agx,agy):
    return L1_approx(agx**2 + agy**2)
    
def Ef(wkg, w_init_g, a):
    """
    :param wkg: single pixel gradient for current watermark estimate for image k
    :param w_init_g: single pixel gradient of initial watermark estimate
    :param a: single pixel alpha
    :return:
    """
    return L1_approx(np.linalg.norm(a*wkg - a*w_init_g,ord=2)**2)
    
def Eaux(w,wk):
    return np.absolute(w - wk)

def L1_approx(s2,e=0.001):
    assert s2 >= 0
    
    return (s2+e**2)**0.5

def add_watermark(W,I, alpha):
    return np.multiply(W,alpha) + np.multiply(I,1-alpha)


def cost(i, w, a, j, igx, igy, agx, agy, wkg, w_init_g, wk):

    cost = Edata(ik,wk,a,jk) + Ereg_ig(igx, igy , agx, agy) + \
           Ereg_ag(agx,agy) + Ef(wkg, w_init_g, a) + Eaux(w,wk)

    return cost


def image_watermark_decomposition(W, W_init, A, Jk):
    """
    :param W: current estimate of global estimate of watermark
    :param W_init:
    :param A: current estimate of matte matrix
    :param Jk: single training image with watermark
    :return:
    """

    Agx = None
    Agy = None
    Ik = copy.deepcopy(Jk)
    Wk = copy.deepcopy(W)
    Ikgx, Ikgy, _ = extract_img_gradients(Ik)
    Wkgx, Wkgy, _ = extract_img_gradients(Wk)
    W_init_gx, W_init_gy, _ = extract_img_gradients(W_init)

    #iterate over individual pixels
    for x in range(Jk.shape[0]):
        for y in range(Jk.shape[1]):
            for n in range(Jk.shape[2]):
                #calculate cost for single pixel in image

                cost_fn = lambda var: cost(var[0], var[1], A[x,y], J[x,y,n],
                                             Ikgx[x,y,n], Ikgy[x,y,n], Agx[x,y,n], Agy[x,y,n],
                                             np.array([Wkgx[x,y,n], Wkgy[x,y,n]]),
                                             np.array([W_init_gx[x, y, n], W_init_gy[x, y, n]]),
                                             Wk[x,y,n])

                #optimize Ik, Wk terms for single pixel
                results = scipy.optimize.minimize(cost_fn, np.array([Ik[x,y,n], Wk[x,y,n]]), method='Newton-CG', maxiter=10)

                Ik[x,y,n] = results.x[0]
                Wk[x,y,n] = results.x[1]


    return Ik, Wk

def matte_update():
    for x in range(Jk.shape[0]):
        for y in range(Jk.shape[1]):
            for n in range(Jk.shape[3]):


