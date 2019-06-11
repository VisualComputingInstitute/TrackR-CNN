import glob
import pycocotools.mask as cocomask
import tensorflow as tf
import numpy as np

from core import Extractions
from datasets import DataKeys
from datasets.Dataset import FileListDataset
from datasets.Loader import register_dataset
from datasets.KITTI.segtrack.KITTI_segtrack import DEFAULT_PATH

NAME = "KITTI_segtrack_bbox_regression_forward"


@register_dataset(NAME)
class KittiSegtrackBboxRegressionForwardDataset(FileListDataset):
  def __init__(self, config, subset):
    super().__init__(config, NAME, subset, DEFAULT_PATH, 2)

  def read_inputfile_lists(self):
    img_filenames = []
    infos = []
    det_files = glob.glob("forwarded/segtrack_lr_0000005/tracking_data/*.txt")
    for det_file in det_files:
      seq = det_file.split("/")[-1].replace(".txt", "")
      with open(det_file) as f:
        for l in f:
          sp = l.split()
          t = int(sp[0])
          img_filename = self.data_dir + "/images/" + seq + "/%06d.png" % t
          img_filenames.append(img_filename)
          infos.append(seq + " " + l)
    return img_filenames, infos

  def load_annotation(self, img, img_filename, ann_string):
    # load mask from rle...
    def _load_mask(ann_str, img_shape):
      ann_str = ann_str.decode("utf-8")
      sp = ann_str.split()
      rle = {"counts": sp[-1], "size": img_shape[:2]}
      m = cocomask.decode(rle)
      return m[..., np.newaxis]
    mask = tf.py_func(_load_mask, [ann_string, tf.shape(img)], tf.uint8)
    mask.set_shape((None, None, 1))
    ann = {DataKeys.SEGMENTATION_LABELS: mask, Extractions.CUSTOM_STRING: ann_string}
    return ann
