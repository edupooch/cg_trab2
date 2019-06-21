import sys

from imutils import paths
import numpy as np

import matplotlib.pyplot as plt
from math import ceil
from cv2 import imread

N_REGIONS_HIST = 4
GLCM_QT_LEVELS = 64


def __init__():
    np.set_printoptions(threshold=sys.maxsize)

    file = open('features.csv', mode='w')
    first = True

    for imagePath in paths.list_images("data"):
        img = imread(imagePath, 0)
        entry = {'name': imagePath.split('\\').pop()}
        features = extract_features(img)
        entry.update(features)
        entry['class'] = imagePath.split('\\')[-2].lower()

        if first:
            print_head_csv(entry, file)
            first = False
        print_features_csv(entry, file=file)

    file.close()


def extract_features(img):
    histogram = get_histogram(img)
    glcm_1_0 = get_glcm(img, (1, 0))
    glcm_0_1 = get_glcm(img, (0, 1))
    glcm_1__1 = get_glcm(img, (1, -1))

    features = {
        "img_avg": get_average(img),
        "img_std": get_st_deviation(img),
        "img_kur": get_kurtosis(img),
        "img_med": get_median(img),

        "hist_0_64_count": get_region_count(histogram[0:64]),
        "hist_0_64_avg": get_average(flat_hist(histogram, 0, 64)),
        "hist_0_64_std": get_st_deviation(flat_hist(histogram, 0, 64)),
        "hist_0_64_kur": get_kurtosis(flat_hist(histogram, 0, 64)),
        "hist_0_64_med": get_median(flat_hist(histogram, 0, 64)),

        "hist_64_128_count": get_region_count(histogram[64:128]),
        "hist_64_128_avg": get_average(flat_hist(histogram, 64, 128)),
        "hist_64_128_std": get_st_deviation(flat_hist(histogram, 64, 128)),
        "hist_64_128_kur": get_kurtosis(flat_hist(histogram, 64, 128)),
        "hist_64_128_median": get_median(flat_hist(histogram, 64, 128)),

        "hist_128_192_count": get_region_count(histogram[128:192]),
        "hist_128_192_avg": get_average(flat_hist(histogram, 128, 192)),
        "hist_128_192_std": get_st_deviation(flat_hist(histogram, 128, 192)),
        "hist_128_192_kur": get_kurtosis(flat_hist(histogram, 128, 192)),
        "hist_128_192_median": get_median(flat_hist(histogram, 128, 192)),

        "hist_192_256_count": get_region_count(histogram[192:256]),
        "hist_192_256_avg": get_average(flat_hist(histogram, 192, 256)),
        "hist_192_256_std": get_st_deviation(flat_hist(histogram, 192, 256)),
        "hist_192_256_kur": get_kurtosis(flat_hist(histogram, 192, 256)),
        "hist_192_256_median": get_median(flat_hist(histogram, 192, 256)),

        "glcm_1_0_asm": get_glcm_asm(glcm_1_0),
        "glcm_1_0_entropy": get_glcm_entropy(glcm_1_0),
        "glcm_1_0_contrast": get_glcm_contrast(glcm_1_0),
        "glcm_1_0_variance": get_glcm_variance(glcm_1_0),
        "glcm_1_0_homogeneity": get_glcm_homogeneity(glcm_1_0),

        "glcm_0_1_asm": get_glcm_asm(glcm_0_1),
        "glcm_0_1_entropy": get_glcm_entropy(glcm_0_1),
        "glcm_0_1_contrast": get_glcm_contrast(glcm_0_1),
        "glcm_0_1_variance": get_glcm_variance(glcm_0_1),
        "glcm_0_1_homogeneity": get_glcm_homogeneity(glcm_0_1),

        "glcm_1__1_asm": get_glcm_asm(glcm_1__1),
        "glcm_1__1_entropy": get_glcm_entropy(glcm_1__1),
        "glcm_1__1_contrast": get_glcm_contrast(glcm_1__1),
        "glcm_1__1_variance": get_glcm_variance(glcm_1__1),
        "glcm_1__1_homogeneity": get_glcm_homogeneity(glcm_1__1),

    }

    return features


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


def get_histogram(img):
    x_size = img.shape[0]
    y_size = img.shape[1]

    histogram = np.zeros(256)
    for i in range(x_size):
        for j in range(y_size):
            gray_level = img[i, j]
            histogram[gray_level] += 1
    return histogram


def flat_hist(histogram, start, end):
    flat = []
    for gray_level in range(start, end):
        frequency = int(histogram[gray_level])
        flat = flat + ([gray_level] * frequency)

    return np.asarray(flat)


def show_histogram(histogram):
    plt.bar(range(256), histogram, width=1, color='black')
    plt.show()


def get_average(obj):
    obj = obj.ravel()
    o_sum = 0

    size = obj.shape[0]

    if size == 0:
        return 0

    for i in range(size):
        o_sum += obj[i]

    return o_sum / size


def get_st_deviation(obj):
    obj = obj.ravel()
    o_sum = 0

    obj_avg = get_average(obj)
    size = obj.shape[0]

    if size == 0:
        return 0

    for i in range(size):
        o_sum += (obj[i] - obj_avg) ** 2

    if size > 1:
        size = size - 1

    return (o_sum / size) ** (1 / 2)


def get_kurtosis(obj):
    obj = obj.ravel()
    o_sum = 0

    obj_avg = get_average(obj)
    obj_std = get_st_deviation(obj)
    size = obj.shape[0]

    if obj_std == 0:
        return 0

    for i in range(size):
        o_sum += (((obj[i] - obj_avg) / obj_std) ** 4)

    return (o_sum / size)


def get_median(obj):
    obj = obj.ravel()
    obj.sort()

    size = obj.shape[0]

    if size == 0:
        return 0
    if size == 1:
        return obj[0]

    middle = size / 2
    is_even = (middle % 2) == 0

    if is_even:
        median = (int(obj[int(middle)]) + int(obj[int(middle + 1)])) / 2
    else:
        median = int(obj[ceil(middle)])

    return median


def get_region_count(histogram_region):
    count = 0
    for i in range(len(histogram_region)):
        count += int(histogram_region[i])
    return count


def get_glcm(img, mask):
    img = reduce_tones(img)

    size_x = img.shape[0]
    size_y = img.shape[1]

    glcm = np.zeros((GLCM_QT_LEVELS, GLCM_QT_LEVELS))
    dis_x = mask[0]
    dis_y = mask[1]
    max_value = 0

    for x in range(size_x):
        for y in range(size_y):
            try:
                point = img[x, y]
                neighbor = img[x + dis_x, y + dis_y]
                glcm[point, neighbor] += 1

                if glcm[point, neighbor] > max_value:
                    max_value = glcm[point, neighbor]
            except:
                pass

    glcm = normalize_glcm(glcm, max_value)
    return glcm


def reduce_tones(img):
    size_x = img.shape[0]
    size_y = img.shape[1]

    for x in range(size_x):
        for y in range(size_y):
            img[x, y] = img[x, y] / (256 / GLCM_QT_LEVELS)

    return img


def normalize_glcm(glcm, max_value):
    for x in range(GLCM_QT_LEVELS):
        for y in range(GLCM_QT_LEVELS):
            glcm[x, y] = glcm[x, y] / max_value
    return glcm


def get_glcm_asm(glcm):
    asm = 0
    for i in range(GLCM_QT_LEVELS):
        for j in range(GLCM_QT_LEVELS):
            asm += glcm[i, j] ** 2
    return asm


def get_glcm_entropy(glcm):
    return get_glcm_asm(glcm) ** (1 / 2)


def get_glcm_contrast(glcm):
    contrast = 0
    for i in range(GLCM_QT_LEVELS):
        for j in range(GLCM_QT_LEVELS):
            contrast += glcm[i, j] * ((i - j) ** 2)
    return contrast


def get_glcm_variance(glcm):
    variance = 0
    for i in range(GLCM_QT_LEVELS):
        for j in range(GLCM_QT_LEVELS):
            variance += glcm[i, j] * abs(i - j)
    return variance


def get_glcm_homogeneity(glcm):
    homogeneity = 0
    for i in range(GLCM_QT_LEVELS):
        for j in range(GLCM_QT_LEVELS):
            homogeneity += glcm[i, j] * (glcm[i, j] / (1 + ((i - j) ** 2)))
    return homogeneity


__init__()
