from imutils import paths
import numpy as np
import cv2
import matplotlib.pyplot as plt


def __init__():
    # for imagePath in paths.list_images("data"):
    #     img = cv2.imread(imagePath, 0)
    #     histogram = get_histogram(img)
    #     show_histogram(histogram)
    #
    #     plt.hist(img.ravel(), 256, [0, 256])
    #     plt.show()

    img = cv2.imread(("data/Capim/RT21_1.jpg"), 0)
    histogram = get_histogram(img)

    his_avg = get_hist_average(histogram)
    img_avg = get_img_average(img)
    # show_histogram(histogram)


def get_histogram(img):
    x_size = img.shape[0]
    y_size = img.shape[1]
    print(x_size, y_size)

    histogram = [0] * 256
    for i in range(x_size):
        for j in range(y_size):
            gray_level = img[i, j]
            histogram[gray_level] += 1
    return histogram


def show_histogram(histogram):
    print(histogram)
    plt.bar(range(256), histogram, width=1, color='black')
    plt.show()


def get_hist_average(histogram):
    hsum = 0
    for value in histogram:
        hsum += value
    return hsum / 256


def get_img_average(img):
    pass


__init__()
