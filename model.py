import json
import math
import utils
import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict


config = utils.load_config()
logger = utils.get_default_logger()


class FCN:
    def __init__(self, images, labels=None, reg=1.0, net_params=None, lr=None):
        self.images = images
        self.labels = labels
        self.reg = reg
        self.lr = lr
        self.params = self.__dict__.copy()
        self.net_params = net_params
        self.build_graph()

    def build_graph(self):
        logger.info('Building model graph...')
        params = self.net_params
        with tf.name_scope('inference'):
            conv1_1 = utils.conv2d(self.images, params['conv1_1'], 'conv1_1')
            conv1_2 = utils.conv2d(conv1_1, params['conv1_2'], 'conv1_2', stride=2)

            conv2_1 = utils.conv2d(conv1_2, params['conv2_1'], 'conv2_1')
            conv2_2 = utils.conv2d(conv2_1, params['conv2_2'], 'conv2_2', stride=2)

            conv3_1 = utils.conv2d(conv2_2, params['conv3_1'], 'conv3_1')
            conv3_2 = utils.conv2d(conv3_1, params['conv3_2'], 'conv3_2', stride=2)

            conv4 = utils.conv2d(conv3_2, 1, 'conv4', kernel_size=(1, 1), activation_fn=None)

        endpoints = OrderedDict()
        endpoints['conv1_1'] = conv1_1
        endpoints['conv1_2'] = conv1_2
        endpoints['conv2_1'] = conv2_1
        endpoints['conv2_2'] = conv2_2
        endpoints['conv3_1'] = conv3_1
        endpoints['conv3_2'] = conv3_2
        endpoints['conv4'] = conv4
        self.endpoints = endpoints

    def train_from_scratch(self, config):
        logger.info('Training from scratch...')
        logger.info(self)
        logger.info('config=\n' + json.dumps(config, indent=2))
        n_steps_per_epoch = int(math.ceil(config['n_examples_for_train'] // config['batch_size']))
        n_steps_for_train = config['n_epochs_for_train'] * n_steps_per_epoch
        train_op, summary_op = self._build_train(config['batch_size'])

        utils.delete_if_exists(config['train_dir'])
        last_loss = slim.learning.train(train_op,
                                        logdir=config['train_dir'],
                                        summary_op=summary_op,
                                        log_every_n_steps=config['log_every'],
                                        save_summaries_secs=config['save_summaries_secs'],
                                        number_of_steps=n_steps_for_train)
        logger.info('Last loss: %.3f' % last_loss)

    def _build_train(self, batch_size):
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        conv4_flatten = tf.reshape(self.endpoints['conv4'], shape=(batch_size, -1), name='conv4_flatten')
        labels_flatten = tf.reshape(self.labels, shape=(batch_size, -1), name='labels_flatten')
        data_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels_flatten,
                                                    logits=conv4_flatten,
                                                    scope='data_loss')
        if self.reg:
            reg_loss = tf.multiply(self.reg,
                                   tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
                                   name='reg_loss')
        else:
            reg_loss = 0.0
        total_loss = tf.add(data_loss, reg_loss, name='total_loss')
        summaries.add(tf.summary.scalar('loss/data_loss', data_loss))
        summaries.add(tf.summary.scalar('loss/total_loss', data_loss))

        global_step = tf.train.get_or_create_global_step()
        solver = tf.train.GradientDescentOptimizer(self.lr)
        train_op = slim.learning.create_train_op(total_loss,
                                                 solver,
                                                 global_step=global_step)

        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        return train_op, summary_op

    def __repr__(self):
        myself = '\n' + '\n'.join('{:>2} {:<10} {}'.format(i, key, value.shape)
                                  for i, (key, value) in enumerate(self.endpoints.items()))
        myself += '\n' + '\n'.join('{:<10}= {}'.format(key, value) for key, value in self.params.items())
        return myself


if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, shape=(1, 400, 600, 3), name='images')
        model = FCN(images, net_params=config['net_params'])
        writer = tf.summary.FileWriter('/tmp/model/', graph=g)
        logger.info(model)
        writer.close()
