# -- coding: utf-8 --
# -*- coding:utf8 -*-

import tensorflow as tf


def liner(argv):
    sess = tf.Session()
    x = tf.Variable(2., tf.float32)
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    model = a * x + b
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(model, {a: 5., b: 9.})
    tf.logging.info(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(liner)
