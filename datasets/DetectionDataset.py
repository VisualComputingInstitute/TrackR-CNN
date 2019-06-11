import tensorflow as tf
import numpy as np
from PIL import Image
from functools import partial

from datasets.Dataset import FileListDataset
from datasets.util.Detection import init_anchors, add_rpn_data, load_instance_seg_annotation_np, load_instance_seg_annotation_from_img_array_np
from datasets import DataKeys
from core.Log import log


class DetectionFileListDataset(FileListDataset):
  def __init__(self, config, dataset_name, subset, default_path, num_classes):
    super().__init__(config, dataset_name, subset, default_path, num_classes)
    self.add_masks = config.bool("add_masks", True)
    self.prefer_gt_to_ignore = config.bool("prefer_gt_to_ignore", False)
    self.use_ioa_for_ignore = config.bool("use_ioa_for_ignore",  False)
    self.use_masks_for_ignore = config.bool("use_masks_for_ignore",  False)
    init_anchors(config)

  def assemble_example(self, tensors):
    tensors = super().assemble_example(tensors)
    if DataKeys.BBOXES_y0x0y1x1 in tensors:
      tensors = add_rpn_data(tensors, self.prefer_gt_to_ignore, use_masks_for_ignore=self.use_masks_for_ignore,
                             use_ioa_for_ignore=self.use_ioa_for_ignore)
    return tensors


class MapillaryLikeDetectionFileListDataset(DetectionFileListDataset):
  def __init__(self, config, dataset_name, subset, default_path, num_classes, n_max_detections,
               class_ids_with_instances, id_divisor, crowd_id=None):
    super().__init__(config, dataset_name, subset, default_path, num_classes)
    self._n_max_detections = n_max_detections
    self._class_ids_with_instances = class_ids_with_instances
    self._id_divisor = id_divisor
    self._crowd_id = crowd_id

    self._preload_dataset = config.bool("preload_dataset", False)
    if self._preload_dataset:
      print("Preloading ENTIRE!!! dataset into memory!!!", file=log.v1)
      imgs, anns = self.read_inputfile_lists()
      self.imgs_preload_dict = {img_filename: np.array(Image.open(img_filename), dtype=np.float32) / 255.0
                                for img_filename in imgs}
      self.anns_preload_dict = {ann_filename: np.array(Image.open(ann_filename)) for ann_filename in anns}
      self._load_ann_np = self._anns_preload_dict_lookup
    else:
      self._load_ann_np = partial(load_instance_seg_annotation_np, n_max_detections=self._n_max_detections,
                                  class_ids_with_instances=self._class_ids_with_instances, id_divisor=self._id_divisor,
                                  crowd_id=self._crowd_id)

  def _anns_preload_dict_lookup(self, ann_filename):
    ann_filename = ann_filename.decode("utf-8")
    ann = self.anns_preload_dict[ann_filename]
    return load_instance_seg_annotation_from_img_array_np(ann, n_max_detections=self._n_max_detections,
                                                          class_ids_with_instances=self._class_ids_with_instances, id_divisor=self._id_divisor,
                                                          crowd_id=self._crowd_id)

  def _imgs_preload_dict_lookup(self, img_filename):
    img_filename = img_filename.decode("utf-8")
    img = self.imgs_preload_dict[img_filename]
    return img

  def load_image(self, img_filename):
    if self._preload_dataset:
      img = tf.py_func(self._imgs_preload_dict_lookup, [img_filename], tf.float32, name="imgs_preload_dict_lookup_np")
      img.set_shape((None, None, 3))
    else:
      img = super(MapillaryLikeDetectionFileListDataset, self).load_image(img_filename)
    return img

  def load_annotation(self, img, img_filename, annotation_filename):
    bboxes, ids, classes, is_crowd, mask = tf.py_func(self._load_ann_np, [annotation_filename],
                                                      [tf.float32, tf.int32, tf.int32, tf.int32, tf.uint8],
                                                      name="postproc_ann_np")
    bboxes.set_shape((self._n_max_detections, 4))
    ids.set_shape((self._n_max_detections,))
    classes.set_shape((self._n_max_detections,))
    is_crowd.set_shape((self._n_max_detections,))
    mask.set_shape((None, None, self._n_max_detections))

    return_dict = {DataKeys.BBOXES_y0x0y1x1: bboxes, DataKeys.CLASSES: classes, DataKeys.IDS: ids,
                   DataKeys.IS_CROWD: is_crowd}
    if self.add_masks:
      return_dict[DataKeys.SEGMENTATION_MASK] = mask
    return_dict = self.postproc_annotation(annotation_filename, return_dict)
    return return_dict
