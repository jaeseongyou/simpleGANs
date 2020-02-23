import tensorflow as tf


def _read_img(data_path):
    img = tf.io.read_file(data_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) * 2. - 1.
    tf.image.central_crop(img, .4)
    img = tf.image.resize(img, [64, 64])
    return img


def make_celeba_dataset():
    data_path = '../dataset/CelebAMask-HQ/CelebA-HQ-img/*.jpg'
    list_ds = tf.data.Dataset.list_files(data_path)
    train_dataset = list_ds.map(_read_img)
    return train_dataset
