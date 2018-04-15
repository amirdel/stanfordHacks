from PIL import Image
import numpy as np
import os
import imgaug as ia
from imgaug import augmenters as iaa

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def myGenerator(file_paths, steps_per_epoch, BATCH_SIZE, INPUT_SHAPE):
    i = 0
    seq = iaa.Sequential([iaa.Sometimes(0.7, iaa.GaussianBlur(sigma=(0, 2.0))),
                      iaa.Sharpen(alpha=(0, 0.1), lightness=(0.7, 1.3)),
                      iaa.ContrastNormalization((0.5, 1.2))],
                     random_order=True)
    img_size = INPUT_SHAPE[0]
    while True:
        x_batch = np.empty((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        y_batch = np.empty((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
        for (ind, j) in enumerate(range(i*BATCH_SIZE, (i+1)*BATCH_SIZE)):
            # pick a random image
            f = np.random.choice(file_paths)
            print(f)
            random_x = np.random.randint(0, 1500-img_size)
            random_y = np.random.randint(0, 1500-img_size)
            xb = np.array(Image.open(f))[random_x:random_x+img_size, random_y:random_y+img_size, :]
            ftruth = f.replace('images', 'labels')
            ftruth = ftruth[:-1]
            yb = np.expand_dims(np.array(Image.open(ftruth))[random_x:random_x+img_size, random_y:random_y+img_size, 0], axis=2)
            yb[yb==255]=1
            if np.random.random() < 0.5:
                xb = flip_axis(xb, 1)
                yb = flip_axis(yb, 1)
            if np.random.random() < 0.5:
                xb = flip_axis(xb, 0)
                yb = flip_axis(yb, 0)
            if np.random.random() < 0.5:
                xb = xb.swapaxes(1, 0)
                yb = yb.swapaxes(1, 0)
            x_batch[ind,...] = xb
            y_batch[ind,...] = yb
        # bunch of augmentation

        x_batch = seq.augment_images(x_batch)
        i += 1
        if i >= steps_per_epoch:
            i = 0
        yield x_batch, y_batch
