import pycocotools.mask as cocomask
import numpy as np
import tensorflow as tf

from core import Extractions
from datasets import DataKeys
from forwarding.Forwarder import Forwarder
from forwarding.tracking.TrackingForwarder_util import export_tracking_result_in_kitti_format, make_disjoint
from forwarding.tracking.Util_tracking import TrackElement


class BboxRefinementForwarder(Forwarder):
  def __init__(self, engine):
    super().__init__(engine)
    self.model_name = self.config.string("model")
    self.mask_disjoint_strategy = self.config.string("mask_disjoint_strategy", "y_pos")  # Or "score"

  def forward(self):
    out_folder = "forwarded/" + self.model_name + "/tracking_data_bbox_refined/"
    tf.gfile.MakeDirs(out_folder)
    for n in range(21):
      open(out_folder + "%04d" % n + ".txt", "w")
    data = self.val_data
    n_examples_per_epoch = data.n_examples_per_epoch()
    extraction_keys = [Extractions.DET_MASKS, DataKeys.IMAGE_FILENAMES, DataKeys.IDS]
    tracks = {}  # tag -> list of lists of trackelems
    for n in range(n_examples_per_epoch):
      res = self.trainer.validation_step(extraction_keys=extraction_keys)
      masks = res[Extractions.EXTRACTIONS][Extractions.DET_MASKS][0][0]
      #if len(masks) > 0:
      #  import matplotlib.pyplot as plt
      #  for mask in masks:
      #    plt.imshow(mask)
      #    plt.show()
      filename = res[Extractions.EXTRACTIONS][DataKeys.IMAGE_FILENAMES][0][0].decode("utf-8")
      sp = filename.split("/")
      seq = sp[-2]
      t = int(sp[-1].replace(".png", "").replace(".jpg", ""))
      ids = res[Extractions.EXTRACTIONS][DataKeys.IDS][0][0]
      masks_encoded = [cocomask.encode(np.asfortranarray(mask)) for mask in masks]
      if seq not in tracks:
        tracks[seq] = []
      while len(tracks[seq]) < t + 1:
        tracks[seq].append([])

      assert len(masks_encoded) == len(ids)
      for id_, mask_ in zip(ids, masks_encoded):
        x0, y0, w, h = cocomask.toBbox(mask_)
        box = [x0, y0, x0 + w, y0 + h]
        obj = data.tracking_result[seq][id_][t]
        class_str = obj.class_
        if class_str == "Car":
          class_id = 1
        elif class_str == "Pedestrian":
          class_id = 2
        else:
          assert False, ("unknown class str", class_str)
        score = obj.score
        tracks[seq][t].append(TrackElement(box=box, track_id=id_, class_=class_id, score=score, mask=mask_))
      print(n, "/", n_examples_per_epoch, masks.shape, filename, ids)

    for seq in tracks.keys():
      tracks[seq] = make_disjoint(tracks[seq], self.mask_disjoint_strategy)
      # write out data
      export_tracking_result_in_kitti_format(seq, tracks[seq], True, self.config.string("model"), out_folder)
