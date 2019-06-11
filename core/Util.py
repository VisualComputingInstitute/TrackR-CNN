import importlib
import pkgutil
import sys

import tensorflow as tf
import numpy as np


def smart_shape(x):
  shape = x.get_shape().as_list()
  tf_shape = tf.shape(x)
  for i, s in enumerate(shape):
    if s is None:
      shape[i] = tf_shape[i]
  return shape


# from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def clip_gradients(grads, threshold):
  gradients, variables = zip(*grads)
  gradients, norm = tf.clip_by_global_norm(gradients, threshold)
  grads = zip(gradients, variables)
  return grads, norm


def calculate_ious(bboxes1, bboxes2):
  # assume layout (x0, y0, x1, y1)
  min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
  max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
  I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
  area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
  area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
  U = area1[:, np.newaxis] + area2[np.newaxis, :] - I
  assert (U > 0).all()
  IOUs = I / U
  assert (IOUs >= 0).all()
  assert (IOUs <= 1).all()
  return IOUs


def calculate_ioas(bboxes1, bboxes2):
  # assume layout (x0, y0, x1, y1)
  min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
  max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
  I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
  area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
  #area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
  A = area1[:, np.newaxis]
  assert (A > 0).all()
  IOAs = I / A
  assert (IOAs >= 0).all()
  assert (IOAs <= 1).all()
  return IOAs


def import_submodules(package_name):
  package = sys.modules[package_name]
  for importer, name, is_package in pkgutil.walk_packages(package.__path__):
    # not sure why this check is necessary...
    if not importer.path.startswith(package.__path__[0]):
      continue
    name_with_package = package_name + "." + name
    importlib.import_module(name_with_package)
    if is_package:
      import_submodules(name_with_package)
