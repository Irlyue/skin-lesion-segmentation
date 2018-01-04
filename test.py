import utils
import inputs
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


from model import FCN


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


if __name__ == '__main__':
    test_two()
