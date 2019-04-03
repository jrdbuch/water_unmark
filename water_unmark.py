from estimate_W import *
from multi_img_matte_and_recon import *
import matplotlib.pyplot as plt
import copy


def main():
    # import training data
    imgs_raw = import_training_images(u'/home/jaredbuchanan/water_unmark/examples/123RF')

    # initial stimate of watermark (correct up to a shift)
    W_init = estimate_watermark(imgs_raw)
    W = copy.deepcopy(W_init)

    # get initial estimate of normalized alpha matte, blend factor, and alpha matter
    A_n = intialize_normalized_alpha_matte(W_init)
    c = 1
    A = A_n * c

    plt.figure(99)
    plt.imshow(A)

    # image watermark decomposition
    Wks = []
    Iks = []
    for Jk in imgs_raw:
        Ik, Wk = image_watermark_decomposition(W, W_init, A, Jk)
        Wks.append(Wk)
        Iks.append(Ik)

        plt.figure(101)
        plt.imshow(Wk)
        plt.figure(102)
        plt.imshow(Ik)


        break



    # generate new global W estimate

    # estimate alpha matte


if __name__ == '__main__':
    #To Do: make this a command line tool

    main()



    plt.show()