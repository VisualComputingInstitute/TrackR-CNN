import glob
import numpy as np
import pycocotools.mask as cocomask
import tensorflow as tf
from scipy.ndimage.morphology import grey_dilation

from datasets import DataKeys
from datasets.KITTI.segtrack.KITTI_segtrack import KittiSegtrackDataset
from datasets.Loader import register_dataset
from forwarding.tracking.TrackingForwarder_util import import_detections_for_sequence

NAME = "KITTI_segtrack_detection_merging"


@register_dataset(NAME)
class KittiSegtrackDetectionMergingDataset(KittiSegtrackDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset)
    self.detections_import_path = config.string("detections_import_path")
    self.detections = self._load_detections()

  def _load_detections(self):
    det_files = glob.glob(self.detections_import_path + "/*.txt")
    dets = {}
    for det_file in det_files:
      tag = det_file.split("/")[-1].replace(".txt", "")
      seq_dets = import_detections_for_sequence(tag, 0, self.detections_import_path, "", 0, True)
      dets[tag] = seq_dets
    return dets

  def load_image(self, img_filename):
    # load detection data instead
    # let's try to present it as if it's an image.
    # important to disable stuff like gamma augmentations then

    # TODO: change this when changing the data encoding
    out_dim = 4

    def _load_and_encode_detections(img_filename_):
      img_filename_ = img_filename_.decode("utf-8")
      sp = img_filename_.split("/")
      tag = sp[-2]
      t = int(sp[-1].replace(".png", "").replace(".jpg", ""))
      boxes, scores, reid, classes, masks = self.detections[tag]
      #boxes = boxes[t]
      scores = scores[t]
      #reid = reid[t]
      classes = classes[t]
      masks = masks[t]
      size = self.img_sizes[tag]
      car_probs = merge_probs(classes, scores, masks, 1, size)
      car_boundary_map = boundary_map(classes, scores, masks, 1, size)
      ped_probs = merge_probs(classes, scores, masks, 2, size)
      ped_boundary_map = boundary_map(classes, scores, masks, 1, size)
      features = np.stack([car_probs, ped_probs, car_boundary_map, ped_boundary_map], axis=2)
      assert features.shape[-1] == out_dim
      return features

    img = tf.py_func(_load_and_encode_detections, [img_filename], tf.float32, name="load_and_encode_detections")
    img.set_shape((None, None, out_dim))
    return {DataKeys.IMAGES: img}


def merge_probs(classes, scores, masks, class_, size):
  h, w = size
  probs = np.zeros((h, w), dtype=np.float32)
  for c, score, mask in zip(classes, scores, masks):
    if c == class_:
      curr_probs = score * cocomask.decode(mask).astype(np.float32)
      probs = np.maximum(curr_probs, probs)
  return probs


def boundary_map(classes, scores, masks, class_, size):
  h, w = size
  res = np.zeros((h, w), dtype=np.float32)
  for c, score, mask in zip(classes, scores, masks):
    if c == class_:
      mask = cocomask.decode(mask).astype(np.float32)
      mask_dil = grey_dilation(mask, size=3)
      boundary = np.logical_and(mask_dil, np.logical_not(mask))
      curr_boundary = np.cast[np.float32](score) * boundary
      res = np.maximum(curr_boundary, res)
  return res
