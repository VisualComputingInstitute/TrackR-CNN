import pycocotools.mask as cocomask
import numpy as np
import tensorflow as tf

from core import Extractions
from datasets import DataKeys
from forwarding.Forwarder import Forwarder
from forwarding.tracking.TrackingForwarder_util import export_detections_for_sequence
from datasets.KITTI.segtrack.KITTI_MOTS_info import SEQ_IDS_VAL


class BboxDetectionRefinementForwarder(Forwarder):
  def __init__(self, engine):
    super().__init__(engine)
    self.model_name = self.config.string("model")

  def forward(self):
    out_folder = "forwarded/" + self.model_name + "/detection_bbox_refined/"
    tf.gfile.MakeDirs(out_folder)
    data = self.val_data
    n_examples_per_epoch = data.n_examples_per_epoch()
    extraction_keys = [Extractions.DET_MASKS, DataKeys.IMAGE_FILENAMES, DataKeys.IDS]
    refined_masks = data.masks
    for n in range(n_examples_per_epoch):
      res = self.trainer.validation_step(extraction_keys=extraction_keys)
      filename = res[Extractions.EXTRACTIONS][DataKeys.IMAGE_FILENAMES][0][0].decode("utf-8")
      sp = filename.split("/")
      seq = sp[-2]
      t = int(sp[-1].replace(".png", "").replace(".jpg", ""))
      ids = res[Extractions.EXTRACTIONS][DataKeys.IDS][0][0]
      masks = res[Extractions.EXTRACTIONS][Extractions.DET_MASKS][0][0]
      print(n + 1, "/", n_examples_per_epoch, filename, len(masks), "boxes/masks")
      masks_encoded = [cocomask.encode(np.asfortranarray(mask)) for mask in masks]

      if seq not in refined_masks or t < 0 or t >= len(refined_masks[seq]):
        assert False, "Mismatch seq or t"
      assert len(masks_encoded) == len(ids)
      for id_, mask_ in zip(ids, masks_encoded):
        if id_ - 1 >= len(refined_masks[seq][t]):
          assert False, "Mismatch id_"
        refined_masks[seq][t][id_ - 1] = mask_

    for seq in SEQ_IDS_VAL:
      # write out data
      export_detections_for_sequence(seq, data.boxes[seq], data.scores[seq], data.reids[seq], data.classes[seq],
                                     refined_masks[seq], "", 0, True, out_folder)
