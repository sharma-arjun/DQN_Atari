"""Loss functions."""

import tensorflow as tf
import semver

def huber_loss(y_true, y_pred, thresh=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    thresh: float, optional
      Positive floating point value. Represents the absolute difference 
      upto which we use the squared error.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    return tf.where(tf.abs(y_true-y_pred) < thresh,0.5*tf.square(y_true-y_pred),tf.abs(y_true-y_pred)-0.5,name='huber_loss')


def mean_huber_loss(y_true, y_pred, thresh=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the absolute difference
      upto which we use the squared loss.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """

    return tf.reduce_mean(tf.where(tf.abs(y_true-y_pred) < thresh,0.5*tf.square(y_true-y_pred),tf.abs(y_true-y_pred)-0.5),name='mean_huber_loss')
