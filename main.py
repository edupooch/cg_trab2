from imutils import paths
import numpy as np

import matplotlib.pyplot as plt
from math import ceil
from cv2 import imread

N_REGIONS_HIST = 4


def print_head_csv(features, file):
    head = ""
    for key in features:
        head = head + str(key) + ","
    print(head[:-1], file=file)


def print_features_csv(features, file):
    entry = ""
    for key in features:
        entry = entry + str(features[key]) + ","
    print(entry[:-1], file=file)


def __init__():
    # for imagePath in paths.list_images("data"):
    #     img = cv2.imread(imagePath, 0)
    #     histogram = get_histogram(img)
    #     show_histogram(histogram)
    #
    #     plt.hist(img.ravel(), 256, [0, 256])
    #     plt.show()

    file = open('features.csv', mode='w')

    img = imread(("data/Capim/RT21_1.jpg"), 0)
    histogram = get_histogram(img)

    features = {
        "img_avg": get_average(img),
        "img_std": get_st_deviation(img),
        "img_kur": get_kurtosis(img),
        "img_med": get_median(img),
        "hist_0_64": get_region_count(histogram, 0, 64),
        "hist_64_128": get_region_count(histogram, 64, 128),
        "hist_128_192": get_region_count(histogram, 128, 192),
        "hist192_256": get_region_count(histogram, 192, 256),
        "his_avg": get_average(histogram),
        "his_std": get_st_deviation(histogram),
        "his_kur": get_kurtosis(histogram),
        "his_med": get_median(histogram),
    }

    print_head_csv(features, file)
    print_features_csv(features, file=file)


def get_histogram(img):
    x_size = img.shape[0]
    y_size = img.shape[1]

    histogram = np.zeros(256)
    for i in range(x_size):
        for j in range(y_size):
            gray_level = img[i, j]
            histogram[gray_level] += 1
    return histogram


def show_histogram(histogram):
    plt.bar(range(256), histogram, width=1, color='black')
    plt.show()


def get_average(obj):
    obj = obj.ravel()
    o_sum = 0

    size = obj.shape[0]

    for i in range(size):
        o_sum += obj[i]

    return o_sum / size


def get_st_deviation(obj):
    obj = obj.ravel()
    o_sum = 0

    obj_avg = get_average(obj)
    size = obj.shape[0]

    for i in range(size):
        o_sum += (obj[i] - obj_avg) ** 2

    return (o_sum / size - 1) ** (1 / 2)


def get_kurtosis(obj):
    obj = obj.ravel()
    o_sum = 0

    obj_avg = get_average(obj)
    obj_std = get_st_deviation(obj)
    size = obj.shape[0]

    for i in range(size):
        o_sum += ((obj[i] - obj_avg) / obj_std) ** 4

    return (o_sum / size)


def get_median(obj):
    obj = obj.ravel()
    obj.sort()

    size = obj.shape[0]
    middle = size / 2

    is_even = (middle % 2) == 0

    if is_even:
        median = (int(obj[int(middle)]) + int(obj[int(middle + 1)])) / 2
    else:
        median = int(obj[ceil(middle)])

    return median


def get_region_count(histogram, start, end):
    count = 0
    for i in range(start,end):
        count += int(histogram[i])
    return count


__init__()
