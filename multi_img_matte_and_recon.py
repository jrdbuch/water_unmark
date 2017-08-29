import numpy as np 
from skimage import io,feature
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
import os
from poisson_reconstruct import poisson_reconstruct
import cv2


def Edata(I,W,a,J):
    #sum over every pixel
    assert I.ndim == 2
    assert W.ndim == 2
    assert J.dim == 2
    
    obj_sum = 0
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            obj_sum += L1_approx(np.absolute(a[x,y]*W[x,y]+(1-a[x,y])*I[x,y]-J[x,y])**2)
            
    return obj_sum
    
def Ereg_IJ(Gx, Gy , ax, ay):
    return L1_approx(np.absolute(ax)*Gx**2 + np.absolute(ay)*Gy**2)

def Ereg_a(ax,ay):
    return L1_approx(ax**2 + ay**2)
    
def Ef(Wm_grad_hat,Wm_grad):
    return L1_approx(np.linalg.norm(Wm_grad_hat - Wm_grad,ord=2)**2)
    
def Eaux(W,Wk):
    obj_sum = 0
    for x in range(W.shape[0]):
        for y in range(W.shape[2]):
            obj_sum += np.absolute(W[x,y] - Wk[x,y])
    
    return obj_sum 

def L1_approx(s2,e=0.001):
    assert s2 >= 0
    
    return (s2+e**2)**0.5


def initialize_vals(Wm_hat,imgs):
    pass

def find_black_patches(img,threshold=0.01):
    pass

def 
    