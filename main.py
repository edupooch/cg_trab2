import sys

from imutils import paths
import numpy as np

import matplotlib.pyplot as plt
from math import ceil
from cv2 import imread, imshow, cv2

IMG_SHAPE = 256
N_REGIONS_HIST = 4
GLCM_QT_LEVELS = 32
N_BINS = 8 + 3


def __init__():
    np.set_printoptions(threshold=sys.maxsize)

    file = open('features.csv', mode='w')
    first = True

    for imagePath in paths.list_images("data"):
        img = imread(imagePath, 0)

        entry = {'name': imagePath.split('/').pop()}

        features = extract_features(img)

        entry.update(features)
        entry['class'] = imagePath.split('/')[-2].lower()

        if first:
            print_head_csv(entry, file)
            first = False
        print_features_csv(entry, file=file)

    file.close()


def extract_features(img):
    histogram = get_histogram(img)
    glcm_3_0 = get_glcm(img, (3, 0))
    glcm_0_3 = get_glcm(img, (0, 3))
    glcm_3__3 = get_glcm(img, (3, -3))

    lbp = get_lbp(img)
    mcc = get_mcc(img)

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

        "glcm_3_0_asm": get_glcm_asm(glcm_3_0),
        "glcm_3_0_entropy": get_glcm_entropy(glcm_3_0),
        "glcm_3_0_contrast": get_glcm_contrast(glcm_3_0),
        "glcm_3_0_variance": get_glcm_variance(glcm_3_0),
        "glcm_3_0_homogeneity": get_glcm_homogeneity(glcm_3_0),

        "glcm_0_3_asm": get_glcm_asm(glcm_0_3),
        "glcm_0_3_entropy": get_glcm_entropy(glcm_0_3),
        "glcm_0_3_contrast": get_glcm_contrast(glcm_0_3),
        "glcm_0_3_variance": get_glcm_variance(glcm_0_3),
        "glcm_0_3_homogeneity": get_glcm_homogeneity(glcm_0_3),

        "glcm_3__3_asm": get_glcm_asm(glcm_3__3),
        "glcm_3__3_entropy": get_glcm_entropy(glcm_3__3),
        "glcm_3__3_contrast": get_glcm_contrast(glcm_3__3),
        "glcm_3__3_variance": get_glcm_variance(glcm_3__3),
        "glcm_3__3_homogeneity": get_glcm_homogeneity(glcm_3__3),
    }

    for i in range(N_BINS):
        key = "lbp_" + str(i)
        features[key] = lbp[i]

    features.update({
        "mcc_rp": get_mcc_rp(mcc),
        "mcc_sre": get_mcc_sre(mcc),
        "mcc_lre": get_mcc_lre(mcc),
        "mcc_gln": get_mcc_gln(mcc),
        "mcc_rln": get_mcc_rln(mcc),
        "mcc_lgre": get_mcc_lgre(mcc),
        "mcc_hgre": get_mcc_hgre(mcc),
        "mcc_srlge": get_mcc_srlge(mcc),
        "mcc_srhge": get_mcc_srhge(mcc),
        "mcc_lrlge": get_mcc_lrlge(mcc),
        "mcc_lrhge": get_mcc_lrhge(mcc), })

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


##### HISTOGRAM #############################

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
    obj = obj.ravel().copy()
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


##### GLCM #############################

def get_glcm(img, mask):
    r_img = reduce_tones(img)
    size_x = r_img.shape[0]
    size_y = r_img.shape[1]

    glcm = np.zeros((GLCM_QT_LEVELS, GLCM_QT_LEVELS))

    dis_x = mask[0]
    dis_y = mask[1]
    max_value = 0

    for x in range(size_x):
        for y in range(size_y):
            if (x + dis_x >= 0 and x + dis_x < size_x) and (y + dis_y >= 0 and y + dis_y < size_y):
                point = int(r_img[x, y])
                neighbor = int(r_img[x + dis_x, y + dis_y])

                glcm[point, neighbor] += 1

                if glcm[point, neighbor] > max_value:
                    max_value = glcm[point, neighbor]

    glcm = normalize_glcm(glcm, max_value)
    return glcm


def reduce_tones(img):
    size_x = img.shape[0]
    size_y = img.shape[1]
    r_img = np.zeros((size_x, size_y))

    for x in range(size_x):
        for y in range(size_y):
            r_img[x, y] = int(img[x, y] / (256 / GLCM_QT_LEVELS))

    return r_img


def normalize_glcm(glcm, max_value):
    norm_glcm = np.zeros((GLCM_QT_LEVELS, GLCM_QT_LEVELS))
    for x in range(GLCM_QT_LEVELS):
        for y in range(GLCM_QT_LEVELS):
            norm_glcm[x, y] = glcm[x, y] / max_value
    return norm_glcm


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


##### LBP #############################
def get_lbp(img):
    lbp = np.zeros(img.shape, dtype=np.uint8)
    img_pad = np.zeros((img.shape[0] + 4, (img.shape[1] + 4)))
    img_pad[2:img.shape[0] + 2, 2:img.shape[1] + 2] = img

    neighbours = [(1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]

    for i in range(0, img_pad.shape[0] - 4):
        for j in range(0, img_pad.shape[0] - 4):
            byte = ""
            for point in neighbours:
                if img_pad[i + 2 + point[0], j + 2 + point[1]] >= img_pad[i + 2, j + 2]:
                    byte = "1" + byte
                else:
                    byte = "0" + byte
            lbp[i, j] = int(byte, 2)

    (hist, _) = np.histogram(lbp.ravel(),
                             bins=N_BINS,
                             range=(0, 8 + 2))
    return hist


##### MCC #############################

def get_mcc(img):
    size_x = img.shape[0]
    size_y = img.shape[1]

    r_img = reduce_tones(img)
    mcc = np.zeros((GLCM_QT_LEVELS, 50), dtype=int)

    for i in range(size_x):
        for j in range(size_y):
            seq = True
            k = 0

            while seq:
                if (j + k < size_y) and (r_img[i, j] == r_img[i, j + k]):
                    k = k + 1
                else:
                    seq = False

            mcc[int(r_img[i, j]), k - 1] += 1

    return mcc


def get_mcc_rp(mcc):
    nr = get_nr(mcc)
    return nr / (IMG_SHAPE * IMG_SHAPE)


def get_nr(mcc):
    nr = 0
    for i in range(mcc.shape[0]):
        for j in range(mcc.shape[1]):
            if mcc[i, j] > 0:
                nr += 1

    return nr


def get_mcc_sre(mcc):
    sre = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            sre += (mcc[i, k] / ((k ** 2) + 1e-7))
    nr = get_nr(mcc)
    return sre / nr


def get_mcc_lre(mcc):
    lre = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            lre += (mcc[i, k] * (k ** 2))

    nr = get_nr(mcc)
    return lre / nr


def get_mcc_gln(mcc):
    gln = 0
    for i in range(mcc.shape[0]):
        gln_k = 0
        for k in range(mcc.shape[1]):
            gln_k += (mcc[i, k])
        gln += gln_k ** 2

    nr = get_nr(mcc)
    return gln / nr


def get_mcc_rln(mcc):
    rln = 0
    for k in range(mcc.shape[1]):
        rln_i = 0
        for i in range(mcc.shape[0]):
            rln_i += (mcc[i, k])
        rln += rln_i ** 2

    nr = get_nr(mcc)
    return rln / nr


def get_mcc_lgre(mcc):
    lgre = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            lgre += (mcc[i, k] / ((i ** 2) + 1e-7))

    nr = get_nr(mcc)
    return lgre / nr


def get_mcc_hgre(mcc):
    hgre = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            hgre += (mcc[i, k] * (i ** 2))

    nr = get_nr(mcc)
    return hgre / nr


def get_mcc_srlge(mcc):
    srlge = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            srlge += (mcc[i, k] / (((i ** 2) * (k ** 2)) + 1e-7))

    nr = get_nr(mcc)
    return srlge / nr


def get_mcc_srhge(mcc):
    srhge = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            srhge += ((mcc[i, k] * (i ** 2)) / ((k ** 2) + 1e-7))

    nr = get_nr(mcc)
    return srhge / nr


def get_mcc_lrlge(mcc):
    lrlge = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            lrlge += ((mcc[i, k] * (k ** 2)) / ((i ** 2) + 1e-7))

    nr = get_nr(mcc)
    return lrlge / nr


def get_mcc_lrhge(mcc):
    lrhge = 0
    for i in range(mcc.shape[0]):
        for k in range(mcc.shape[1]):
            lrhge += (mcc[i, k] * (i ** 2) * (k ** 2))

    nr = get_nr(mcc)
    return lrhge / nr


__init__()
