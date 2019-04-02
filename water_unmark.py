from estimate_W import *
from multi_img_matte_and_recon import *
import copy


def main():
    # import training data
    imgs_raw = import_training_images(u'/home/jaredbuchanan/water_unmark/examples/123RF')

    # estimate watermark (correct up to a shift)
    W_init = estimate_watermark(imgs_raw)
    W = copy.deepcopy(W_init)

    # get initial estimate for alpha matte
    A = None

    # image watermark decomposition
    Wks = []
    Iks = []
    for Jk in imgs_raw:
        Ik, Wk = image_watermark_decomposition(W, W_init, A, Jk)
        Wks.append(Wk)
        Iks.apend(Ik)

    # generate new global W estimate

    # estimate alpha matte



if __name__ == '__main__':
    #To Do: make this a command line tool

    main()



    plt.show()