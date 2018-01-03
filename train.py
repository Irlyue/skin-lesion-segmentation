import utils
import model
import inputs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from scipy.misc import imresize


logger = utils.get_default_logger()


def prep(data):
    one, two = [], []
    images, labels = data.images, data.labels
    for a, b in zip(images, labels):
        one.append(imresize(a, size=(400, 400)))
        label = imresize(b, size=(50, 50), interp='nearest')
        label[label == 255] = 1
        two.append(label)
    data.images = np.asarray(one, dtype=np.float32)
    data.labels = np.asarray(two, dtype=np.int32)
    return data


def train_small():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = utils.load_config()
    with tf.Graph().as_default():
        dermis = inputs.SkinData(config['data_dir'], 'dermis')
        dermis = prep(dermis)
        image, label = dermis.images[0], dermis.labels[0]
        images, labels = tf.constant(image[None], dtype=tf.float32), tf.constant(label[None], dtype=tf.float32)
        net = model.FCN(images, labels,
                        net_params=config['net_params'],
                        lr=config['learning_rate'])
        net.train_from_scratch(config)


def test_small():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = utils.load_config()
    with tf.Graph().as_default():
        dermis = inputs.SkinData(config['data_dir'], 'dermis')
        dermis = prep(dermis)
        image, label = dermis.images[0], dermis.labels[0]
        images, labels = tf.constant(image[None], dtype=tf.float32), tf.constant(label[None], dtype=tf.float32)
        global_step = tf.train.get_or_create_global_step()
        net = model.FCN(images, labels,
                        net_params=config['net_params'])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(config['train_dir']))
            logger.info('Model-%d restored successfully!' % sess.run(global_step))

            logits = np.squeeze(sess.run(net.endpoints['conv4']))
            out = np.zeros_like(logits, dtype=np.uint8)
            out[logits > 0] = 1
            correct = np.array(label == out, dtype=np.float32)
            accuracy = np.mean(correct)
            logger.info('accuracy: %.3f' % accuracy)
            plt.subplot(121)
            plt.imshow(label, cmap='gray')
            plt.subplot(122)
            plt.imshow(out, cmap='gray')
            plt.show()


def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = utils.load_config()
    with tf.Graph().as_default():
        dermis = inputs.SkinData(config['data_dir'], 'dermis')
        dermis = prep(dermis)
        images, labels = tf.constant(dermis.images), tf.constant(dermis.labels)
        batch_images, batch_labels = tf.train.batch([images, labels],
                                                    enqueue_many=True,
                                                    batch_size=config['batch_size'])
        net = model.FCN(batch_images, batch_labels,
                        net_params=config['net_params'],
                        lr=config['learning_rate'])
        net.train_from_scratch(config)


if __name__ == '__main__':
    test_small()
