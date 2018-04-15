import numpy as np
import os
import imgaug as ia
from imgaug import augmenters as iaa

def myGenerator(file_paths, steps_per_epoch, BATCH_SIZE, INPUT_SHAPE):
    i = 0
    while True:
        x_batch = np.empty((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        y_batch = np.empty((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], 1))
        for (ind, j) in enumerate(range(i*BATCH_SIZE, (i+1)*BATCH_SIZE)):
            # pick a random image
            f = np.random.choice(file_paths)
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
                xb = flip_axis(xb, 2)
                yb = flip_axis(yb, 2)
            if np.random.random() < 0.5:
                xb = xb.swapaxes(1, 2)
                yb = yb.swapaxes(1, 2)
                
            x_batch[ind,...] = xb
            y_batch[ind,...] = yb
        # bunch of augmentation
        seq = iaa.Sequential([iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),
                              iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                              iaa.ContrastNormalization((0.75, 1.2)),
                              iaa.ContrastNormalization((0.5, 1.0))],
                             random_order=True)
        x_batch = seq.augment_images(x_batch)
        i += 1
        if i >= steps_per_epoch:
            i = 0
        yield x_batch, y_batch
