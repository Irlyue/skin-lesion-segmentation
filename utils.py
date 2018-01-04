import os
import time
import json
import tarfile
import logging
import importlib
import logging.config
import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_config(path=None):
    path = 'config.json' if path is None else path
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def load_module(name):
    return importlib.import_module(name)


def get_logger(name):
    log = logging.getLogger(name)
    return log


DEFAULT_LOGGER = None


def get_default_logger():
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = logging.getLogger('ALL')
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s'))

        DEFAULT_LOGGER.setLevel(logging.DEBUG)
        DEFAULT_LOGGER.addHandler(handler)
    return DEFAULT_LOGGER


def delete_if_exists(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def class_wise_pooling(x, m, scope='class_pool'):
    """
    Operation for class-wise pooling.

    :param x: Tensor, with shape(batch_size, h, w, m*c)
    :param m: int, parameter M in the paper
    :param scope: str, parameter scope
    :return:
        op: Tensor, with shape(batch_size, h, w, c)
    """
    with tf.variable_scope(scope):
        _, _, _, n = x.get_shape().as_list()
        n_classes = n // m
        ops = []
        for i in range(n_classes):
            class_avg_op = tf.reduce_mean(x[:, :, :, m*i:m*(i+1)], axis=3, keep_dims=True)
            ops.append(class_avg_op)
        final_op = tf.concat(ops, axis=3)
        return final_op


def spatial_pooling(x, k, alpha=None, scope='spatial_pool'):
    """
    Operation for spatial pooling.

    :param x: Tensor, with shape(batch_size, h, w, c)
    :param k: int,
    :param alpha: float, mixing coefficient for kmax and kmin. If none, ignore kmin.
    :param scope: str, parameter scope
    :return:
        op: Tensor, with shape(batch_size, c)
    """
    with tf.variable_scope(scope):
        batch_size, _, _, n_classes = x.get_shape().as_list()
        x_flat = tf.reshape(x, shape=(batch_size, -1, n_classes))
        x_transp = tf.transpose(x_flat, perm=(0, 2, 1))
        k_maxs, _ = tf.nn.top_k(x_transp, k, sorted=False)
        k_maxs_mean = tf.reduce_mean(k_maxs, axis=2)
        result = k_maxs_mean
        if alpha:
            # top -x_flat to retrieve the k smallest values
            k_mins, _ = tf.nn.top_k(-x_transp, k, sorted=False)
            # flip back
            k_mins = -k_mins
            k_mins_mean = tf.reduce_mean(k_mins, axis=2)
            alpha = tf.constant(alpha, name='alpha', dtype=tf.float32)
            result += alpha * k_mins_mean
        return result


def extract_to(src, dst):
    with tarfile.open(os.path.expanduser(src)) as f:
        f.extractall(dst)


def conv2d(inputs,
           n_filters,
           scope=None,
           stride=1,
           wd=1e-3,
           padding='SAME',
           kernel_size=(3, 3),
           activation_fn=tf.nn.relu):
    return slim.conv2d(inputs,
                       scope=scope,
                       num_outputs=n_filters,
                       padding=padding,
                       stride=stride,
                       kernel_size=kernel_size,
                       activation_fn=activation_fn,
                       weights_regularizer=slim.l2_regularizer(wd))


class Timer:
    def __init__(self):
        self._tic = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.eclipsed = time.time() - self._tic