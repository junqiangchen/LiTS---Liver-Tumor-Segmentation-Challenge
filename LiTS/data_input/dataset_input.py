#coding=utf-8

import pandas as pd
import tensorflow as tf
import random

preprocessing_dict = {'resize_shape':[512, 512],
                      'rotate':True,
                      'rotate_fix':True,
                      'flip':True,
                      'brightness':True,
                      'brightness_range':0.2,
                      'saturation':True,
                      'saturation_range':[0.5, 1.5],
                      'contrast':True,
                      'contrast_range':[0.5, 1.5]}

image_type = 'jpg'

#TODO how to add image_type and preprocessing_dict as addition arg in map function

def _parse_function(image, mask):
    image_string = tf.read_file(image)
    mask_string = tf.read_file(mask)
    if image_type == 'jpg':
        image_decoded = tf.image.decode_jpeg(image_string, 0)
        mask_decoded = tf.image.decode_jpeg(mask_string, 1)
    elif image_type == 'png':
        image_decoded = tf.image.decode_png(image_string, 0)
        mask_decoded = tf.image.decode_png(mask_string, 1)
    elif image_type == 'bmp':
        image_decoded = tf.image.decode_bmp(image_string, 0)
        mask_decoded = tf.image.decode_bmp(mask_string, 1)
    else:
        raise TypeError('==> Error: Only support jpg, png and bmp.')
        
    # already in 0~1
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    mask_decoded = tf.image.convert_image_dtype(mask_decoded, tf.float32)
    
    return image_decoded, mask_decoded

def _preprocess_function(image_decoded, mask_decoded):  
    shape = preprocessing_dict['resize_shape']
    assert len(shape) == 2 and isinstance(shape, list), '==> Error: shape error.'
    image = tf.image.resize_images(image_decoded, shape)
    mask = tf.image.resize_images(mask_decoded, shape)

    # randomly rotate
    if preprocessing_dict['rotate'] ==  True:
        if preprocessing_dict['rotate_fix'] == True:
            k = random.sample([1,2,3], 1)[0]
            image = tf.image.rot90(image, k)
            mask = tf.image.rot90(mask, k)
        else:
            raise ValueError('==> Error: Only support rotate 90, 180 and 270 degree.')

    # randomly flip
    if preprocessing_dict['flip'] ==  True:
        k = [1, 2]
        if random.sample(k, 1) == [1]:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        else:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

    # adjust the brightness of images by a random factor
    if preprocessing_dict['brightness'] == True:
        delta = preprocessing_dict['brightness']
        # delta randomly picked in the interval [-delta, delta)
        image = tf.image.random_brightness(image, max_delta=delta)

    # adjust the saturation of an RGB image by a random factor
    if preprocessing_dict['saturation'] == True:
        saturation_range = preprocessing_dict['saturation_range']
        assert len(saturation_range) == 2 and isinstance(saturation_range, list), '==> Error: saturation_range error.'
        image = tf.image.random_saturation(image, *saturation_range)

    # adjust the contrast of an image by a random factor
    if preprocessing_dict['contrast'] == True:
        contrast_range = preprocessing_dict['contrast_range']
        assert len(contrast_range) == 2 and isinstance(contrast_range, list), '==> Error: saturation_range error.'
        image = tf.image.random_contrast(image, *contrast_range)

    # make sure pixel value in 0~1
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

def datagenerator(imagecsv_path, maskcsv_path, batch_size):
    """
    return: data iterator
    """
    df_image = pd.read_csv(imagecsv_path)
    df_mask = pd.read_csv(maskcsv_path)

    try:
        image_filenames = tf.constant(df_image['filename'].tolist())
        mask_filenames = tf.constant(df_mask['filename'].tolist())
    except:
        raise ValueError('==> csv error')

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(_preprocess_function)
    dataset = dataset.batch(batch_size)
    data_iterator = dataset.make_initializable_iterator()

    return data_iterator

def main():
    # test
    import cv2
    import numpy as np

    data_iterator = datagenerator('trainX.csv', 'trainY.csv', 1)
    with tf.Session() as sess:
        sess.run(data_iterator.initializer)
        next_batch = data_iterator.get_next()
        image, mask = sess.run(next_batch)
        cv2.imwrite('testimage.jpg', cv2.cvtColor(np.squeeze(image) * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite('testmask.jpg', np.squeeze(mask) * 255)

if __name__ == '__main__':
    main()
