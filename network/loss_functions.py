import numpy as np
import tensorflow as tf


def huber_loss(X, Y):
    err = X - Y
    loss = tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)
    loss = tf.reduce_sum(loss)

    return loss


def mse_loss(X, Y):
    err = X - Y
    return tf.reduce_sum(tf.square(err))

def kl_loss(y_true,y_pred):
    return tf.reduce_sum(y_true*(tf.log(y_true/y_pred)))



