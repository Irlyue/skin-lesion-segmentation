import os
import utils
import inputs
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim


from model import FCN
from inputs import image_prep_for_test
from crf import crf_post_process
from testing_utils import metric_accuracy


logger = utils.get_default_logger()


def test():
    config = utils.load_config()
    with tf.Graph().as_default():
        with tf.device('/cpu'):
            dermis = inputs.SkinData(config['data_dir'], 'dermis')
            global_step = tf.train.get_or_create_global_step()
            image_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
            label_ph = tf.placeholder(dtype=tf.float32, shape=(None, None))
            images = tf.expand_dims(image_ph, axis=0)
            labels = tf.expand_dims(label_ph, axis=0)
            labels = tf.expand_dims(labels, axis=-1)
            net = FCN(images,
                      net_params=config['net_params'])
            h, w = tf.shape(label_ph)[0], tf.shape(label_ph)[1]
            output = tf.image.resize_nearest_neighbor(net.endpoints['out'], size=(h, w))
            correct = tf.cast(tf.equal(labels, output), dtype=tf.float32)
            correct_count = tf.reduce_sum(correct)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(config['train_dir']))
            logger.info('Model at step-%d restored successfully!' % sess.run(global_step))
            total_pixels = 0
            correct_pixels = 0
            for i, (image, label) in enumerate(zip(dermis.images, dermis.labels)):
                if i % 10 == 0:
                    logger.info('Processing image %i...' % i)
                correct_pixels += sess.run(correct_count, feed_dict={image_ph: image / 255.0, label_ph: label})
                total_pixels += np.product(label.shape)
            accuracy = correct_pixels * 1.0 / total_pixels
            logger.info('Accuracy: %.3f' % accuracy)


def test_two():
    config = utils.load_config()
    dermis = inputs.SkinData(config['data_dir'], 'dermis')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        batch_images, batch_labels = dermis.test_batch()
        net = FCN(batch_images,
                  net_params=config['net_params'])

        name_to_values, name_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_accuracy(predictions=net.outputs,
                                                        labels=batch_labels)
        })
        results = slim.evaluation.evaluate_once('',
                                                tf.train.latest_checkpoint(config['train_dir']),
                                                '/tmp/model',
                                                num_evals=config['n_examples_for_train'],
                                                eval_op=list(name_to_updates.values()),
                                                final_op=list(name_to_values.values()))

        for key, value in zip(name_to_values.keys(), results):
            logger.info('{:<10} {}'.format(key, value))


def test_three():
    config = utils.load_config()
    dermis = inputs.SkinData(config['data_dir'], 'dermis')
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        image_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
        images = tf.expand_dims(image_ph, axis=0)
        net = FCN(images,
                  net_params=config['net_params'])

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(config['train_dir']))
            logger.info('Model at step-%d restored successfully!' % sess.run(global_step))
            utils.create_if_not_exists(config['save_path'])
            for i, (image, label) in enumerate(zip(dermis.images, dermis.labels)):
                if i % 5 == 0:
                    logger.info('Processing image %d...' % i)

                prep_image = image_prep_for_test(image)
                pred = np.squeeze(sess.run(net.outputs, feed_dict={image_ph: prep_image}))
                path = os.path.join(config['save_path'], dermis.listing[i].split('/')[-1] + '.jpg')
                save_all(image, label, pred, path)


def test_four():
    config = utils.load_config()
    dermis = inputs.SkinData(config['data_dir'], 'dermis')
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        image_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
        images = tf.expand_dims(image_ph, axis=0)
        net = FCN(images,
                  net_params=config['net_params'])

        h, w = tf.shape(image_ph)[0], tf.shape(image_ph)[1]
        upscore = tf.image.resize_images(net.endpoints['conv4'], size=(h, w))
        prob_one = tf.nn.sigmoid(upscore)
        prob_zero = 1 - prob_one
        probs = tf.concat([prob_zero, prob_one], axis=3)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(config['train_dir']))
            logger.info('Model at step-%d restored successfully!' % sess.run(global_step))
            utils.create_if_not_exists(config['save_path'])
            total_count = 0
            true_count = 0
            for i, (image, label) in enumerate(zip(dermis.images, dermis.labels)):
                prep_image = image_prep_for_test(image)
                probs_o = np.squeeze(sess.run(probs, feed_dict={image_ph: prep_image}))
                cnn_result = np.argmax(probs_o, axis=2)
                accuracy_before, _, _ = metric_accuracy(cnn_result, label)
                cnn_crf_result = crf_post_process(image, probs_o)
                accuracy_i, true_count_i, total_count_i = metric_accuracy(cnn_crf_result, label)
                true_count += true_count_i
                total_count += total_count_i

                ss = 'DOWN' if accuracy_before > accuracy_i else 'UP'
                path = os.path.join(config['save_path'], dermis.listing[i].split('/')[-1])
                path = '{:<6} ({:.3f}) ({:.3f}) {}.jpg'.format(path, accuracy_before, accuracy_i, ss)
                save_all_two(image, label, cnn_result, cnn_crf_result, path)

                if i % 5 == 0:
                    logger.info('Image-%d accuracy before(%.3f) after(%.3f) %s' % (i, accuracy_before, accuracy_i, ss))

            accuracy = true_count * 1.0 / total_count
            logger.info('Accuracy after crf: %.3f' % accuracy)


def save_all(image, label, pred, path):
    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(label, cmap='gray')
    plt.subplot(133)
    plt.imshow(pred, cmap='gray')
    plt.savefig(path)


def save_all_two(image, label, cnn_result, cnn_crf_result, path):
    plt.subplot(221)
    plt.imshow(image)
    plt.subplot(222)
    plt.imshow(label, cmap='gray')
    plt.subplot(223)
    plt.imshow(cnn_result, cmap='gray')
    plt.subplot(224)
    plt.imshow(cnn_crf_result, cmap='gray')
    plt.savefig(path)


if __name__ == '__main__':
    # test_two()
    test_four()
