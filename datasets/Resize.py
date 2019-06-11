from enum import Enum, unique
from functools import partial
import tensorflow as tf
import numpy as np

from datasets import DataKeys
from datasets.util.Util import resize_image, random_crop_image, resize_coords_yxyx, resize_coords_xy, \
  get_crop_offset


@unique
class ResizeMode(Enum):
  FIXED_SIZE = "fixed_size"
  UNCHANGED = "unchanged"
  RANDOM_RESIZE_AND_CROP = "random_resize_and_crop"
  BBOX_CROP_AND_RESIZE_FIXED_SIZE = "bbox_crop_and_resize_fixed_size"
  RESIZE_MIN_SHORT_EDGE_MAX_LONG_EDGE = "resize_min_short_edge_max_long_edge"
  FIXED_RESIZE_AND_CROP = "fixed_resize_and_crop"
  FIXED_PERCENTAGE = "fixed_percentage"


def resize(tensors, resize_mode, size):
  if resize_mode == ResizeMode.FIXED_SIZE:
    return resize_fixed_size(tensors, size)
  elif resize_mode == ResizeMode.FIXED_PERCENTAGE:
    return resize_fixed_percentage(tensors, size)
  elif resize_mode == ResizeMode.UNCHANGED:
    return resize_unchanged(tensors)
  elif resize_mode == ResizeMode.RANDOM_RESIZE_AND_CROP:
    return random_resize_and_crop(tensors, size)
  elif resize_mode == ResizeMode.FIXED_RESIZE_AND_CROP:
    return fixed_resize_and_crop(tensors, size)
  elif resize_mode == ResizeMode.BBOX_CROP_AND_RESIZE_FIXED_SIZE:
    return bbox_crop_and_resize_fixed_size(tensors, size)
  elif resize_mode == ResizeMode.RESIZE_MIN_SHORT_EDGE_MAX_LONG_EDGE:
    return resize_min_short_edge_max_long_edge(tensors, size)
  else:
    assert False, ("resize mode not implemented yet", resize_mode)


def jointly_resize(tensors_batch, resize_mode, size):
  if resize_mode == ResizeMode.FIXED_SIZE or resize_mode == ResizeMode.FIXED_PERCENTAGE or\
    resize_mode == ResizeMode.UNCHANGED or resize_mode == ResizeMode.RESIZE_MIN_SHORT_EDGE_MAX_LONG_EDGE:
    return [resize(tensor, resize_mode, size) for tensor in tensors_batch]
  elif resize_mode == ResizeMode.RANDOM_RESIZE_AND_CROP:
    return joint_random_resize_and_crop(tensors_batch, size)
  elif resize_mode == ResizeMode.FIXED_RESIZE_AND_CROP or resize_mode == ResizeMode.BBOX_CROP_AND_RESIZE_FIXED_SIZE:
    assert False, ("resize mode makes no sense in joint mode", resize_mode)
  else:
    assert False, ("resize mode not implemented yet", resize_mode)


def resize_fixed_size(tensors, size):
  tensors_resized = tensors.copy()
  orig_img_size = tf.shape(tensors[DataKeys.IMAGES])[:2]

  class Resizing_:
    Bilinear, NN, Coords_yxyx, Coords_xy = range(4)

  def resize_(key_, resizing_):
    if key_ in tensors:
      val = tensors[key_]
      if resizing_ == Resizing_.Bilinear:
        resized = resize_image(val, size, True)
      elif resizing_ == Resizing_.NN:
        resized = resize_image(val, size, False)
      elif resizing_ == Resizing_.Coords_yxyx:
        resized = resize_coords_yxyx(val, size, orig_img_size)
      elif resizing_ == Resizing_.Coords_xy:
        resized = resize_coords_xy(val, size, orig_img_size)
      else:
        assert False
      tensors_resized[key_] = resized

  keys_to_resize_bilinear = [DataKeys.IMAGES]
  keys_to_resize_nn = [DataKeys.SEGMENTATION_LABELS, DataKeys.BBOX_GUIDANCE, DataKeys.RAW_SEGMENTATION_LABELS,
                       DataKeys.SEGMENTATION_MASK, DataKeys.SEGMENTATION_INSTANCE_LABELS]
  keys_to_resize_coords_yxyx = [DataKeys.BBOXES_y0x0y1x1]
  keys_to_resize_coords_xy = []
  for key in keys_to_resize_bilinear:
    resize_(key, Resizing_.Bilinear)
  for key in keys_to_resize_nn:
    resize_(key, Resizing_.NN)
  for key in keys_to_resize_coords_yxyx:
    resize_(key, Resizing_.Coords_yxyx)
  for key in keys_to_resize_coords_xy:
    resize_(key, Resizing_.Coords_xy)
  return tensors_resized


def resize_fixed_percentage(tensors, size):
  assert len(size) == 1
  percentage = float(size[0])/100.0
  h = tf.cast(tf.shape(tensors[DataKeys.IMAGES])[0], tf.float32)
  w = tf.cast(tf.shape(tensors[DataKeys.IMAGES])[1], tf.float32)

  newh = tf.multiply(percentage, h)
  neww = tf.multiply(percentage, w)

  newh = tf.cast(tf.round(newh), tf.int32)
  neww = tf.cast(tf.round(neww), tf.int32)

  return resize_fixed_size(tensors, tf.stack([newh, neww], axis=0))


def resize_min_short_edge_max_long_edge(tensors, size):
  # Reference: CustomResize._get_augment_params in
  # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/common.py
  assert len(size) == 2
  min_short_edge = float(size[0])
  max_long_edge = float(size[1])
  h = tf.cast(tf.shape(tensors[DataKeys.IMAGES])[0], tf.float32)
  w = tf.cast(tf.shape(tensors[DataKeys.IMAGES])[1], tf.float32)

  scale = min_short_edge * 1.0 / tf.minimum(h, w)
  less_val = tf.less(h, w)
  newh = tf.where(less_val, min_short_edge, tf.multiply(scale, h))
  neww = tf.where(less_val, tf.multiply(scale, w), min_short_edge)

  scale = max_long_edge * 1.0 / tf.maximum(newh, neww)
  greater_val = tf.greater(tf.maximum(newh, neww), max_long_edge)
  newh = tf.where(greater_val, tf.multiply(scale, newh), newh)
  neww = tf.where(greater_val, tf.multiply(scale, neww), neww)

  newh = tf.cast(tf.round(newh), tf.int32)
  neww = tf.cast(tf.round(neww), tf.int32)

  return resize_fixed_size(tensors, tf.stack([newh, neww], axis=0))


def resize_unchanged(tensors):
  if DataKeys.SEGMENTATION_LABELS in tensors:
    tensors[DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE] = tensors[DataKeys.SEGMENTATION_LABELS]
  return tensors


def random_resize_and_crop(tensors, size):
  assert len(size) in (1, 2)
  if len(size) == 2:
    assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size, size]
  tensors, _, _ = resize_random_scale_with_min_size(tensors, min_size=crop_size)
  tensors, _ = random_crop_tensors(tensors, crop_size)
  return tensors


def joint_random_resize_and_crop(tensors_batch, size):
  assert len(size) in (1, 2)
  if len(size) == 2:
    assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size, size]

  img = tensors_batch[0][DataKeys.IMAGES]
  img_h = tf.shape(img)[0]
  img_w = tf.shape(img)[1]
  scaled_size, scale_factor = get_resize_random_scale_with_min_size_factor(img_h, img_w, min_size=crop_size)

  resized_tensors = []
  for tensors in tensors_batch:
    tensors = resize_fixed_size(tensors, scaled_size)
    resized_tensors.append(tensors)

  crop_offset = get_crop_offset(scaled_size, crop_size)
  resized_and_cropped_tensors = []
  for tensors in resized_tensors:
    tensors, crop_offset = random_crop_tensors(tensors, crop_size, crop_offset)
    resized_and_cropped_tensors.append(tensors)

  return resized_and_cropped_tensors


def fixed_resize_and_crop(tensors, size):
  assert len(size) in (1, 2)
  if len(size) == 2:
    assert size[0] == size[1]
    crop_size = size
  else:
    crop_size = [size, size]
  tensors = scale_with_min_size(tensors, min_size=crop_size)
  tensors = object_crop_fixed_offset(tensors, crop_size)

  return tensors


def random_crop_tensors(tensors, size, offset=None):
  tensors_cropped = tensors.copy()

  keys_to_crop = [DataKeys.IMAGES, DataKeys.SEGMENTATION_LABELS, DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE,
                  DataKeys.BBOX_GUIDANCE, DataKeys.RAW_SEGMENTATION_LABELS,
                  DataKeys.SEGMENTATION_INSTANCE_LABELS]
  # offset = None

  def _crop(key_, offset_):
    if key_ in tensors:
      val = tensors[key_]
      cropped, offset_ = random_crop_image(val, size, offset_)
      tensors_cropped[key_] = cropped
    return offset_

  for key in keys_to_crop:
    offset = _crop(key, offset)

  return tensors_cropped, offset


def object_crop_fixed_offset(tensors, size):
  label = tensors[DataKeys.SEGMENTATION_LABELS]
  object_locations = tf.cast(tf.where(tf.not_equal(label, 0))[:, :2], tf.int32)
  min_val = tf.maximum(tf.constant([0, 0]),
                       tf.reduce_max(object_locations, axis=0) - size)
  offset = tf.concat([min_val, [0]], axis=0)

  return random_crop_tensors(tensors, size, offset)[0]


def bbox_crop_and_resize_fixed_size(tensors, size):
  MARGIN = 50
  if len(size) > 2:
    MARGIN = size[2]
    size = size[:2]
  tensors_cropped = tensors.copy()

  assert DataKeys.BBOXES_y0x0y1x1 in tensors
  bbox = tensors[DataKeys.BBOXES_y0x0y1x1]
  bbox_rounded = tf.cast(tf.round(bbox), tf.int32)
  y0, x0, y1, x1 = tf.unstack(bbox_rounded)

  # add margin and clip to bounds
  shape = tf.shape(tensors[DataKeys.IMAGES])
  y0 = tf.maximum(y0 - MARGIN, 0)
  x0 = tf.maximum(x0 - MARGIN, 0)
  y1 = tf.minimum(y1 + MARGIN, shape[0])
  x1 = tf.minimum(x1 + MARGIN, shape[1])

  def crop_and_resize(key_, bilinear):
    if key_ in tensors:
      val = tensors[key_]
      res = val[y0:y1, x0:x1]
      res = resize_image(res, size, bilinear)
      tensors_cropped[key_] = res

  keys_to_resize_bilinear = [DataKeys.IMAGES]
  keys_to_resize_nn = [DataKeys.SEGMENTATION_LABELS, DataKeys.BBOX_GUIDANCE, DataKeys.RAW_SEGMENTATION_LABELS]

  for key in keys_to_resize_bilinear:
    crop_and_resize(key, True)

  for key in keys_to_resize_nn:
    crop_and_resize(key, False)

  if DataKeys.LASER_GUIDANCE in tensors:
    laser = tensors[DataKeys.LASER_GUIDANCE][y0: y1, x0: x1]
    laser = resize_laser_to_fixed_size(laser, size)
    tensors_cropped[DataKeys.LASER_GUIDANCE] = laser

  if DataKeys.SEGMENTATION_LABELS in tensors:
    tensors_cropped[DataKeys.SEGMENTATION_LABELS_ORIGINAL_SIZE] = \
      tensors[DataKeys.SEGMENTATION_LABELS]

  tensors_cropped[DataKeys.CROP_BOXES_y0x0y1x1] = tf.stack([y0, x0, y1, x1])

  if DataKeys.BBOXES_x0y0x1y1 in tensors:
    # Here bboxes_x0y0x1y1 is a different bbox than used for cropping. We use it for bbox regression
    # Transform the box according to crop and resize into the new coordinates
    x02, y02, x12, y12 = tf.unstack(tensors[DataKeys.BBOXES_x0y0x1y1])
    x02 -= tf.cast(x0, tf.float32)
    x12 -= tf.cast(x0, tf.float32)
    y02 -= tf.cast(y0, tf.float32)
    y12 -= tf.cast(y0, tf.float32)
    #x02 = tf.Print(x02, [x02], message="x02")
    #y02 = tf.Print(y02, [y02], message="y02")
    #x12 = tf.Print(x12, [x12], message="x12")
    #y12 = tf.Print(y12, [y12], message="y12")
    h = tf.cast(y1 - y0, tf.float32)
    w = tf.cast(x1 - x0, tf.float32)
    x0_new = x02 * size[1] / w
    x1_new = x12 * size[1] / w
    y0_new = y02 * size[0] / h
    y1_new = y12 * size[0] / h
    tensors_cropped[DataKeys.BBOXES_x0y0x1y1] = tf.stack([x0_new, y0_new, x1_new, y1_new])

    #tensors_cropped[DataKeys.BBOXES_x0y0x1y1] = tf.Print(tensors_cropped[DataKeys.BBOXES_x0y0x1y1], [tensors_cropped[DataKeys.BBOXES_x0y0x1y1]], message="box_after_resize", summarize=100)

  return tensors_cropped


def get_resize_random_scale_with_min_size_factor(img_h, img_w, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  shorter_side = tf.minimum(img_h, img_w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  min_scale = tf.maximum(min_scale, min_scale_factor)
  max_scale = tf.maximum(max_scale, min_scale_factor)
  scale_factor = tf.random_uniform(shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast([img_h, img_w], tf.float32) * scale_factor), tf.int32)
  return scaled_size, scale_factor


def resize_random_scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors[DataKeys.IMAGES]
  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  scaled_size, scale_factor = get_resize_random_scale_with_min_size_factor(h, w, min_size, min_scale, max_scale)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out, scaled_size, scale_factor


def scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors[DataKeys.IMAGES]

  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  shorter_side = tf.minimum(h, w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast(tf.shape(img)[:2], tf.float32) * min_scale_factor), tf.int32)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def resize_laser_to_fixed_size(laser, size):
  f = partial(resize_laser_to_fixed_size_np, size=size)
  laser = tf.py_func(f, [laser], tf.float32, name="resize_laser_to_fixed_size")
  laser.set_shape(size + [1])
  return laser


def resize_laser_to_fixed_size_np(laser, size):
  # here we assume, that 1 is used for foreground, -1 for background and 0 for "no reading"
  fg_y, fg_x, _ = (laser == 1).nonzero()
  bg_y, bg_x, _ = (laser == -1).nonzero()

  def scale_indices(ind, size_in, size_out):
    return (np.round(((ind + 0.5) * size_out / size_in) - 0.5)).astype(np.int)

  fg_y = scale_indices(fg_y, laser.shape[0], size[0])
  fg_x = scale_indices(fg_x, laser.shape[1], size[1])
  bg_y = scale_indices(bg_y, laser.shape[0], size[0])
  bg_x = scale_indices(bg_x, laser.shape[1], size[1])

  laser_out = np.zeros(size + [1], dtype=np.float32)
  laser_out[fg_y, fg_x, 0] = 1
  laser_out[bg_y, bg_x, 0] = -1

  return laser_out
