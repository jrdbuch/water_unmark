from estimate_W import *



if __name__ == '__main__':
    #To Do: make this a command line tool

    imgs_raw = import_training_images(u'/home/jaredbuchanan/water_unmark/examples/123RF')
    w_estimate = estimate_watermark(imgs_raw)


    plt.show()