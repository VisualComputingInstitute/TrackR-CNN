import tensorflow as tf
import numpy as np

from datasets import DataKeys
from datasets.KITTI.segtrack.KITTI_segtrack import KittiSegtrackDetectionDataset
from datasets.Loader import register_dataset
from datasets.util.TrackingGT import load_tracking_det_KITTI

NAME = "KITTI_segtrack_bbox_refinement"


@register_dataset(NAME)
class KittiSegtrackBboxRefinementDataset(KittiSegtrackDetectionDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME)
    self.tracking_result_to_refine_path = config.string("tracking_result_to_refine_path")
    self.tracking_result = load_tracking_det_KITTI(self.tracking_result_to_refine_path,
                                                   filter_to_cars_and_pedestrians=True)

  def load_annotation(self, img, img_filename, annotation_filename):
    def _load_boxes_to_refine(ann_filename):
      ann_filename = ann_filename.decode("utf-8")
      seq = ann_filename.split("/")[-2]
      t = int(ann_filename.split("/")[-1].replace(".png", ""))
      bboxes_to_refine = np.zeros((self._n_max_detections, 4), dtype="float32")
      ids = np.zeros((self._n_max_detections,), dtype="int32")
      idx = 0
      for track in self.tracking_result[seq].values():
        if t in track:
          obj = track[t]
          bboxes_to_refine[idx] = obj.bbox_x0y0x1y1
          ids[idx] = obj.id_
          idx += 1
      return bboxes_to_refine[:idx], ids[:idx]
    boxes_to_refine, ids = tf.py_func(_load_boxes_to_refine, [annotation_filename],
                                      [tf.float32, tf.int32], name="load_boxes_to_refine")
    boxes_to_refine.set_shape((None, 4))
    ids.set_shape((None,))
    return_dict = {DataKeys.BBOXES_TO_REFINE_x0y0x1y1: boxes_to_refine,
                   DataKeys.IMAGE_FILENAMES: img_filename,
                   DataKeys.IDS: ids}
    return return_dict
