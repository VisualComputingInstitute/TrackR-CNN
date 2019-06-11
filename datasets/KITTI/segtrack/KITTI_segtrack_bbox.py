import tensorflow as tf
import numpy as np

from datasets import DataKeys
from datasets.Mapillary.MapillaryLike_instance import MapillaryLikeInstanceDataset
from datasets.util.TrackingGT import load_tracking_gt_KITTI
from datasets.KITTI.segtrack.KITTI_segtrack import DEFAULT_PATH
from datasets.Loader import register_dataset
from datasets.util.Util import username

NAME = "KITTI_segtrack_bbox_regression"
KITTI_TRACKING_GT_DEFAULT_PATH = "/home/" + username() + "/vision/mask_annotations/KITTI_tracking_annotations/"


@register_dataset(NAME)
class KittiSegtrackBboxRegressionDataset(MapillaryLikeInstanceDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME, DEFAULT_PATH, "datasets/KITTI/segtrack/", 1000, [1, 2])
    self._gt = load_tracking_gt_KITTI(KITTI_TRACKING_GT_DEFAULT_PATH, filter_to_cars_and_pedestrians=True)

  def postproc_annotation(self, ann_filename, ann):
    ann = super().postproc_annotation(ann_filename, ann)
    if not isinstance(ann, dict):
      ann = {DataKeys.SEGMENTATION_LABELS: ann}

    def _lookup_bbox(fn):
      fn = fn.decode("utf-8")
      sp = fn.split(":")
      id_ = int(sp[-1]) % 1000
      sp = sp[0].split("/")
      t = int(sp[-1].replace(".png", ""))
      seq = sp[-2]
      seq_gt = self._gt[seq]

      if id_ in seq_gt and t in seq_gt[id_]:
        id_gt = seq_gt[id_]
        obj = id_gt[t]
        bbox_x0y0x1y1 = obj[3].astype("float32")
        return False, bbox_x0y0x1y1
      else:
        return True, np.zeros((4,), dtype=np.float32)

    bbox_is_invalid, bbox = tf.py_func(_lookup_bbox, [ann_filename], [tf.bool, tf.float32], name="lookup_bbox")
    bbox_is_invalid.set_shape(())
    bbox.set_shape((4,))
    ann[DataKeys.SKIP_EXAMPLE] = bbox_is_invalid
    ann[DataKeys.BBOXES_x0y0x1y1] = bbox
    return ann
