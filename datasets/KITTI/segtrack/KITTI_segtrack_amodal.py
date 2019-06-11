import numpy as np
import tensorflow as tf

from datasets import DataKeys
from datasets.KITTI.segtrack.KITTI_segtrack import KittiSegtrackDataset
from datasets.Loader import register_dataset

NAME = "KITTI_segtrack_amodal"


@register_dataset(NAME)
class KittiSegtrackAmodalDataset(KittiSegtrackDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME)

  def postproc_annotation(self, ann_filename, ann):
    ann = super().postproc_annotation(ann_filename, ann)
    if not isinstance(ann, dict):
      ann = {DataKeys.SEGMENTATION_LABELS: ann}

    def _lookup_amodal_bbox(fn):
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

    amodal_bbox_is_invalid, amodal_bbox = tf.py_func(_lookup_amodal_bbox, [ann_filename], [tf.bool, tf.float32],
                                                     name="lookup_amodal_bbox")
    amodal_bbox_is_invalid.set_shape(())
    amodal_bbox.set_shape((4,))

    # TODO: implement this correctly
    assert False, "not implemented yet"
    ann[DataKeys.BBOXES_x0y0x1y1] = bbox
    return ann
