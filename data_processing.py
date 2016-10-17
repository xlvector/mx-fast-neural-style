import numpy as np
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle
import logging
import random
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

def VggMean():
    return [123.68, 116.779, 103.939]

def PreprocessContentImage(path, short_edge, dshape=None):
    img = io.imread(path)
    logging.info('content img %s with shape %s', path, img.shape)
    if len(img.shape) != 3:
        return None
    factor = float(short_edge) / min(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    if dshape != None:
        # random crop
        xx = int((sample.shape[0] - dshape[2]))
        yy = int((sample.shape[1] - dshape[3]))
        xstart = random.randint(0, xx)
        ystart = random.randint(0, yy)
        xend = xstart + dshape[2]
        yend = ystart + dshape[3]
        sample = sample[xstart:xend, ystart:yend, :]

    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    vgg_mean = VggMean()
    for k in range(3):
        sample[k, :] -= vgg_mean[k]
    # logging.info("resize the content image to %s", sample.shape)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreprocessStyleImage(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    
    vgg_mean = VggMean()
    for k in range(3):
        sample[k, :] -= vgg_mean[k]
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostprocessImage(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    print img
    vgg_mean = VggMean()
    for k in range(3):
        img[k, :] += vgg_mean[k]
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename, remove_noise=0.05):
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=remove_noise, multichannel=True)
    io.imsave(filename, out)




