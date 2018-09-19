import numpy as np
from keras.preprocessing import image  # 按照特定的矩阵对图片进行转换


def RGB_PCA(images):
    pixels = images.reshape(-1, images.shape[-1])
    idx = np.random.random_integers(0, pixels.shape[0], 1000000)
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels) / 256.
    C = np.cov(pixels) / (256. * 256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image, eig_val, eig_vec):
    a = np.random.randn(3)
    v = np.array([a[0] * eig_val[0], a[1] * eig_val[1], a[2] * eig_val[2]])
    variation = np.dot(eig_vec, v)
    return image + variation


# pca白化
def img_pca(img):
    l, v, m = RGB_PCA(img)
    img = RGB_variations(img, l, v)
    return img

