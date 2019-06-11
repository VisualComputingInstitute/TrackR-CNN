import tensorflow as tf
import numpy as np

from datasets import DataKeys
from datasets.KITTI.segtrack.KITTI_segtrack import KittiSegtrackDetectionDataset
from datasets.KITTI.segtrack.KITTI_MOTS_info import SEQ_IDS_VAL, TIMESTEPS_PER_SEQ
from datasets.Loader import register_dataset
from forwarding.tracking.TrackingForwarder_util import import_detections_for_sequence

NAME = "KITTI_segtrack_bbox_detection_refinement"


@register_dataset(NAME)
class KittiSegtrackBboxDetectionRefinementDataset(KittiSegtrackDetectionDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME)
    self.detections_to_refine_path = config.string("detections_to_refine_path")
    self.boxes, self.scores, self.reids, self.classes, self.masks = {}, {}, {}, {}, {}
    for seq in SEQ_IDS_VAL:
      self.boxes[seq], self.scores[seq], self.reids[seq], self.classes[seq], self.masks[seq] =\
        import_detections_for_sequence(seq, TIMESTEPS_PER_SEQ[seq], self.detections_to_refine_path, "", 0, True)

  def load_annotation(self, img, img_filename, annotation_filename):
    def _load_boxes_to_refine(ann_filename):
      ann_filename = ann_filename.decode("utf-8")
      seq = ann_filename.split("/")[-2]
      t = int(ann_filename.split("/")[-1].replace(".png", ""))
      bboxes_to_refine = np.zeros((self._n_max_detections, 4), dtype="float32")
      ids = np.zeros((self._n_max_detections,), dtype="int32")
      idx = 0
      for box in self.boxes[seq][t]:
        bboxes_to_refine[idx] = box
        ids[idx] = idx + 1
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
