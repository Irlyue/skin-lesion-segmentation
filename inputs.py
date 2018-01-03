import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from scipy.misc import imread


def get_image_list(data_dir, db):
    path_one = os.path.join(data_dir, 'Skin Image Data Set-1/skin_data/melanoma/')
    path_two = os.path.join(data_dir, 'Skin Image Data Set-2/skin_data/notmelanoma/')
    melanoma = [os.path.join(path_one, db, item.split('_')[0])
                for item in os.listdir(os.path.join(path_one, db)) if not item.endswith('db')]
    not_melanoma = [os.path.join(path_two, db, item.split('_')[0])
                    for item in os.listdir(os.path.join(path_two, db)) if not item.endswith('db')]
    return melanoma, not_melanoma


def load_one_database(data_dir, db):
    images = []
    labels = []
    melanoma, not_melanoma = get_image_list(data_dir, db)
    all_files = melanoma + not_melanoma
    for item in all_files:
        image_path = item + '_orig.jpg'
        label_path = item + '_contour.png'
        image = imread(image_path)
        label = imread(label_path)
        label[label == 255] = 1
        images.append(image)
        labels.append(label)
    return images, labels, all_files


class SkinData:
    def __init__(self, data_dir, db):
        self.db = db
        self.images, self.labels, self.listing = load_one_database(data_dir, db)

    def show_shapes(self):
        for i in range(len(self.images)):
            print('{:<7} shape{}'.format(self.listing[i].split('/')[-1], self.images[i].shape))

    def __repr__(self):
        myself = '\n'
        myself += 'database:  %s\n' % self.db
        myself += '#examples: %s\n' % len(self.images)
        return myself

    def data_batch(self, batch_size, input_size):
        """
        Generate a input batch.

        :param batch_size: int, batch size
        :param input_size: tuple or list, input dimension for the model
        :return:
        """
        image_list = [item + '_orig.jpg' for item in self.listing]
        label_list = [item + '_contour.png' for item in self.listing]
        image_files, label_files = tf.convert_to_tensor(image_list), tf.convert_to_tensor(label_list)
        queue = tf.train.slice_input_producer([image_files, label_files],
                                              shuffle=True)
        img_contents = tf.read_file(queue[0])
        label_contents = tf.read_file(queue[1])
        image = tf.image.decode_jpeg(img_contents, channels=3)
        label = tf.image.decode_png(label_contents, channels=1)
        image, label = default_image_prep(image, label, input_size)
        return tf.train.batch([image, label],
                              batch_size=batch_size)


def default_image_prep(image, label, input_size):
    crop_h, crop_w = input_size
    image, label = image_scaling(image, label)
    image, label = image_mirroring(image, label)
    image, label = random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w)

    # normalization
    image = image / 255.0
    return image, label


def image_scaling(img, label):
    """
    Randomly scales the images between 0.9 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """

    scale = tf.random_uniform([1], minval=1.0, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    return img, label


def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop


if __name__ == '__main__':
    dermis = SkinData('/home/wenfeng/Desktop/Image-Segmentations/icpr-2018/data', 'dermis')
    print(dermis)
    with tf.Graph().as_default():
        rp = '/home/wenfeng/Desktop/Image-Segmentations/icpr-2018/data'
        data = SkinData(rp, 'dermis')
        images, labels = data.data_batch(1, (400, 400))
        with tf.train.MonitoredSession() as sess:
            image, label = sess.run([images, labels])
            print(image.shape, label.shape)
            print(image)
            plt.subplot(121)
            plt.imshow(image[0].astype(np.uint8))
            plt.subplot(122)
            plt.imshow(label[0, :, :, 0], cmap='gray')
            plt.show()
    # dermquest = SkinData('/home/wenfeng/Desktop/Image-Segmentations/icpr-2018/data', 'dermquest')
    # print(dermquest)
    # dermquest.show_shapes()

