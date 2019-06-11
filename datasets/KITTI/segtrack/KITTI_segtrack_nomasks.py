# This dataset is like KITTI segtrack but uses the originally annotated tracking bounding boxes and no masks.
# The purpose of this dataset is just to be used for an ablation experiment.
import tensorflow as tf
import numpy as np

from datasets import DataKeys
from datasets.KITTI.segtrack.KITTI_segtrack import KittiSegtrackDataset
from datasets.Loader import register_dataset
from datasets.util.TrackingGT import load_tracking_gt_KITTI

NAME = "KITTI_segtrack_nomasks"
BBOX_GT_PATH = "/globalwork/voigtlaender/data/KITTI/training/label_02/"


@register_dataset(NAME)
class KittiSegtrackNoMaskDataset(KittiSegtrackDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, name=NAME)
    self.gt = load_tracking_gt_KITTI(BBOX_GT_PATH, filter_to_cars_and_pedestrians=False)

  def load_annotation(self, img, img_filename, annotation_filename):
    def _load_ann(ann_filename):
      ann_filename = ann_filename.decode("utf-8")
      # they need to be padded to n_max_detections
      bboxes = np.zeros((self._n_max_detections, 4), dtype="float32")
      ids = np.zeros(self._n_max_detections, dtype="int32")
      classes = np.zeros(self._n_max_detections, dtype="int32")
      is_crowd = np.zeros(self._n_max_detections, dtype="int32")
      idx = 0

      seq = ann_filename.split("/")[-2]
      t = int(ann_filename.split("/")[-1].replace(".png", ""))
      for track in self.gt[seq].values():
        if t in track:
          obj = track[t]
          ids[idx] = obj.id_ + 1
          if obj.class_ == "Car":
            classes[idx] = 1
          elif obj.class_ == "Pedestrian":
            classes[idx] = 2
          else:
            ids[idx] = 10000
            is_crowd[idx] = 1
            classes[idx] = 10
          x0, y0, x1, y1 = obj.bbox_x0y0x1y1
          bboxes[idx] = np.array([y0, x0, y1, x1])
          assert idx < self._n_max_detections
          idx += 1
      return bboxes, ids, classes, is_crowd

    bboxes, ids, classes, is_crowd = tf.py_func(_load_ann, [annotation_filename],
                                                [tf.float32, tf.int32, tf.int32, tf.int32],
                                                name="postproc_ann_np")
    bboxes.set_shape((self._n_max_detections, 4))
    ids.set_shape((self._n_max_detections,))
    classes.set_shape((self._n_max_detections,))
    is_crowd.set_shape((self._n_max_detections,))

    return_dict = {DataKeys.BBOXES_y0x0y1x1: bboxes, DataKeys.CLASSES: classes, DataKeys.IDS: ids,
                   DataKeys.IS_CROWD: is_crowd}
    return return_dict
