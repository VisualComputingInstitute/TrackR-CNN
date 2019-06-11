import tensorflow as tf
import pickle
import numpy as np

from core.Util import calculate_ious
from datasets import DataKeys
from datasets.Dataset import FileListDataset
from datasets.Loader import register_dataset
from datasets.util.BoundingBox import get_bbox_from_segmentation_mask_np

NAME = "mapillary_crop"
DEFAULT_PATH = "/globalwork/voigtlaender/data/mapillary/"


@register_dataset(NAME, resolution="quarter")
@register_dataset(NAME + "_full", resolution="full")
@register_dataset(NAME + "_half", resolution="half")
@register_dataset(NAME + "_quarter", resolution="quarter")
class MapillaryCropDataset(FileListDataset):
  def __init__(self, config, subset, resolution):
    self.resolution = resolution
    assert resolution in ("quarter", "half", "full"), resolution
    if resolution == "full":
      default_path = DEFAULT_PATH
    else:
      default_path = DEFAULT_PATH.replace("/mapillary/", "/mapillary_{}/".format(resolution))
    super().__init__(config, NAME, subset, default_path, 2)
    det_data_path = config.string("det_data_path", "/globalwork/voigtlaender/data/mapillary_dets/Faster-R50C4/")
    with open(det_data_path + "/dets_" + subset + ".pkl", "rb") as f:
      self._det_data = pickle.load(f)
    self._instance_ids = {}
    self._data_list_path = "datasets/Mapillary/"
    self._id_divisor = 256
    self.proposal_sampling_exponent = config.float("proposal_sampling_exponent", 0.0)

  def read_inputfile_lists(self):
    data_list = "training.txt" if self.subset == "train" else "validation.txt"
    data_list = self._data_list_path + "/" + data_list
    imgs = []
    anns = []
    with open(data_list) as f:
      for l in f:
        im, an, *im_ids_and_sizes = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an
        found = False
        for id_and_size in im_ids_and_sizes:
          id_ = id_and_size.split(":")[0]
          #size_ = int(id_and_size.split(":")[1])
          cat_id = int(id_) // self._id_divisor
          # class 19 person
          # TODO: 20, 21, 22 could be used as ignore classes
          if cat_id == 19 and len(self._det_data[im.split("/")[-1]]) > 0:
            found = True
            k = im.split("/")[-1].replace(".jpg", "")
            if k not in self._instance_ids:
              self._instance_ids[k] = set()
            self._instance_ids[k].add(int(id_))
        if found:
          imgs.append(im)
          anns.append(an)
    return imgs, anns

  def load_annotation(self, img, img_filename, annotation_filename):
    ann_data = tf.read_file(annotation_filename)
    ann = tf.image.decode_image(ann_data, dtype=tf.uint16, channels=1)
    ann.set_shape(img.get_shape().as_list()[:-1] + [1])
    ann = self.postproc_annotation(annotation_filename, ann)
    return ann

  def postproc_annotation(self, ann_filename, ann):
    class_, bbox, mask = tf.py_func(self._postproc_annotation, [ann_filename, ann], [tf.int64, tf.float32, tf.uint8])
    class_.set_shape(())
    bbox.set_shape((4,))
    mask.set_shape((None, None, 1))
    return {DataKeys.CLASSES: class_, DataKeys.BBOXES_y0x0y1x1: bbox, DataKeys.SEGMENTATION_LABELS: mask}

  def _postproc_annotation(self, ann_filename, ann):
    ann_filename = ann_filename.decode("utf-8")
    dets = self._det_data[ann_filename.split("/")[-1].replace(".png", ".jpg")]
    scores = dets[:, 4]
    # sample one
    probs = scores ** self.proposal_sampling_exponent
    probs /= probs.sum()
    idx = np.random.choice(dets.shape[0], p=probs)
    det = dets[idx]
    box = det[:4]
    if self.resolution == "half":
      box *= 2
    elif self.resolution == "full":
      box *= 4

    gt_ids = self._instance_ids[ann_filename.split("/")[-1].replace(".png", "")]
    # get gt masks
    gt_masks = [(ann == id_).astype(np.uint8) for id_ in gt_ids]
    # get gt boxes
    gt_boxes = np.array([get_bbox_from_segmentation_mask_np(mask) for mask in gt_masks], np.float32)
    # change to (x0, y0, x1, y1)
    gt_boxes = gt_boxes[:, [1, 0, 3, 2]]
    ious = calculate_ious(box[np.newaxis], gt_boxes)[0]
    iou = ious.max()
    max_idx = ious.argmax()
    if iou > 0.5:
      class_ = np.cast[np.int64](1)
      mask = gt_masks[max_idx]
    else:
      class_ = np.cast[np.int64](0)
      mask = np.full_like(ann, 255, dtype=np.uint8)

    # shuffle to y0x0y1x1
    box_y0x0y1x1 = box[[1, 0, 3, 2]]
    return class_, box_y0x0y1x1, mask
