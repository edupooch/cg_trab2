import sys

from imutils import paths
import numpy as np

import matplotlib.pyplot as plt
from math import ceil
from cv2 import imread, imshow, cv2

IMG_SHAPE = 256
N_REGIONS_HIST = 4
GLCM_QT_LEVELS = 32


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
    glrm = get_glrm(img)

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

    for i in range(len(lbp)):
        key = "lbp_" + str(i)
        features[key] = lbp[i]

    features.update({
        "glrm_rp": get_glrm_rp(glrm),
        "glrm_sre": get_glrm_sre(glrm),
        "glrm_lre": get_glrm_lre(glrm),
        "glrm_gln": get_glrm_gln(glrm),
        "glrm_rln": get_glrm_rln(glrm),
        "glrm_lgre": get_glrm_lgre(glrm),
        "glrm_hgre": get_glrm_hgre(glrm),
        "glrm_srlge": get_glrm_srlge(glrm),
        "glrm_srhge": get_glrm_srhge(glrm),
        "glrm_lrlge": get_glrm_lrlge(glrm),
        "glrm_lrhge": get_glrm_lrhge(glrm), })

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
            if (0 <= x + dis_x < size_x) and (0 <= y + dis_y < size_y):
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

    hist = get_histogram(lbp)

    UNIFORM_PATTERNS = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112,
                        120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225,
                        227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255]

    NOT_UNIFORM = [i for i in range(256) if i not in UNIFORM_PATTERNS]

    lbp_hist = np.zeros(len(UNIFORM_PATTERNS) + 1, dtype=int)

    for i in range(len(UNIFORM_PATTERNS)):
        lbp_hist[i] = hist[UNIFORM_PATTERNS[i]]

    for i in range(len(NOT_UNIFORM)):
        lbp_hist[len(UNIFORM_PATTERNS)] += hist[NOT_UNIFORM[i]]

    # LBP_HIST_SIZE = 16
    # lbp_hist = np.zeros(LBP_HIST_SIZE, dtype=int)
    # for i in range(len(hist)):
    #     index = int(i / (256 / LBP_HIST_SIZE))
    #     lbp_hist[index] += hist[i]

    return lbp_hist


##### glrm #############################

def get_glrm(img):
    size_x = img.shape[0]
    size_y = img.shape[1]

    r_img = reduce_tones(img)
    glrm = np.zeros((GLCM_QT_LEVELS, 50), dtype=int)

    for i in range(size_x):
        for j in range(size_y):
            seq = True
            k = 0

            while seq:
                if (j + k < size_y) and (r_img[i, j] == r_img[i, j + k]):
                    k = k + 1
                else:
                    seq = False

            glrm[int(r_img[i, j]), k - 1] += 1

    return glrm


def get_glrm_rp(glrm):
    nr = get_nr(glrm)
    return nr / (IMG_SHAPE * IMG_SHAPE)


def get_nr(glrm):
    nr = 0
    for i in range(glrm.shape[0]):
        for j in range(glrm.shape[1]):
            if glrm[i, j] > 0:
                nr += 1

    return nr


def get_glrm_sre(glrm):
    sre = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            sre += (glrm[i, k] / ((k ** 2) + 1e-7))
    nr = get_nr(glrm)
    return sre / nr


def get_glrm_lre(glrm):
    lre = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            lre += (glrm[i, k] * (k ** 2))

    nr = get_nr(glrm)
    return lre / nr


def get_glrm_gln(glrm):
    gln = 0
    for i in range(glrm.shape[0]):
        gln_k = 0
        for k in range(glrm.shape[1]):
            gln_k += (glrm[i, k])
        gln += gln_k ** 2

    nr = get_nr(glrm)
    return gln / nr


def get_glrm_rln(glrm):
    rln = 0
    for k in range(glrm.shape[1]):
        rln_i = 0
        for i in range(glrm.shape[0]):
            rln_i += (glrm[i, k])
        rln += rln_i ** 2

    nr = get_nr(glrm)
    return rln / nr


def get_glrm_lgre(glrm):
    lgre = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            lgre += (glrm[i, k] / ((i ** 2) + 1e-7))

    nr = get_nr(glrm)
    return lgre / nr


def get_glrm_hgre(glrm):
    hgre = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            hgre += (glrm[i, k] * (i ** 2))

    nr = get_nr(glrm)
    return hgre / nr


def get_glrm_srlge(glrm):
    srlge = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            srlge += (glrm[i, k] / (((i ** 2) * (k ** 2)) + 1e-7))

    nr = get_nr(glrm)
    return srlge / nr


def get_glrm_srhge(glrm):
    srhge = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            srhge += ((glrm[i, k] * (i ** 2)) / ((k ** 2) + 1e-7))

    nr = get_nr(glrm)
    return srhge / nr


def get_glrm_lrlge(glrm):
    lrlge = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            lrlge += ((glrm[i, k] * (k ** 2)) / ((i ** 2) + 1e-7))

    nr = get_nr(glrm)
    return lrlge / nr


def get_glrm_lrhge(glrm):
    lrhge = 0
    for i in range(glrm.shape[0]):
        for k in range(glrm.shape[1]):
            lrhge += (glrm[i, k] * (i ** 2) * (k ** 2))

    nr = get_nr(glrm)
    return lrhge / nr


__init__()
