# -- coding: utf-8 --
# -*- coding:utf8 -*-

import tensorflow as tf


def main(argv):
    sess = tf.Session()
    say_hello(sess)
    sess.close()


def say_hello(sess):
    hello = tf.placeholder(tf.string)
    world = tf.placeholder(tf.string)
    model = hello + ' ' + world + '!'
    result = sess.run(model, {hello: 'Hello', world: 'world'})
    tf.logging.info(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
