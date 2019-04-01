import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct
import cv2


def Edata(i,w,alpha,j):
    """
    Cost associated with the deviation of W applied to I via the formation model (Jform) as compared with the actual J

    :param i: Natural Image, single pixel
    :param w: Watermark image, single pixel
    :param a: alpha matte, single pixel
    :param j: natural image with watermark, single pixel
    :return:
    """

    jform = add_watermark(w, i, alpha)
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

def Ereg_ag(ax,ay):
    return L1_approx(ax**2 + ay**2)
    
def Ef(Wm_grad_hat,Wm_grad):
    return L1_approx(np.linalg.norm(Wm_grad_hat - Wm_grad,ord=2)**2)
    
def Eaux(w,wk):
    return np.absolute(w - wk)

def L1_approx(s2,e=0.001):
    assert s2 >= 0
    
    return (s2+e**2)**0.5

def add_watermark(W,I, alpha):
    return np.multiply(W,alpha) + np.multiply(I,1-alpha)


def cost():
    cost = Edata() + Ereg_ig() + Ereg_ag() + Ef() + Eaux()

    return cost


def image_watermark_decomposition(W, alpha, Jk):
    """
    :param W: current estimate of global watermark
    :param alpha: current estimate of alpha
    :param Jk: single image sample
    :return:
    """

    alpha_x = None
    alpha_y = None

    for x in range(Jk.shape[0]):
        for y in range(Jk.shape[1]):
            for n in range(Jk.shape[3]):
                #calculate cost for single pixel in image

                cost_fn = lambda x: cost_fn()

                #optimize ix, wx terms


    return

def matte_update():
    for x in range(Jk.shape[0]):
        for y in range(Jk.shape[1]):
            for n in range(Jk.shape[3]):


