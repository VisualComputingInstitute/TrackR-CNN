import tensorflow as tf
import numpy as np
import glob

from core.Util import calculate_ious
from datasets import DataKeys
from datasets.Dataset import FileListDataset
from datasets.Loader import register_dataset
from datasets.util.TrackingGT import load_tracking_gt_mot

NAME = "MOT_crop"
DEFAULT_PATH = "/globalwork/voigtlaender/data/MOTS_challenge/train/images/"


@register_dataset(NAME)
class MotCropDataset(FileListDataset):
  def __init__(self, config, subset):
    super().__init__(config, NAME, subset, DEFAULT_PATH, 2)
    gt_file_pattern = self.config.string("mot_gt_path", "/globalwork/gnana/mot/gt_MOT_train_dpm/*.txt")
    gt_files = glob.glob(gt_file_pattern)
    self._gt = load_tracking_gt_mot(None, False, gt_files)

    #mot_det_path = self.config.string("mot_det_path", "/globalwork/gnana/mot/MOT17/train/")
    mot_det_path = self.config.string("mot_det_path", "/home/voigtlaender/vision/motchallenge-devkit/res/MOT17Det/FRCNN/data/clean/")
    self._dets = {}
    for seq in self._gt:
      #det_filename = mot_det_path + "/MOT17-" + seq[-2:] + "-FRCNN/det/det.txt"
      det_filename = mot_det_path + "/MOT17-" + seq[-2:] + ".txt"
      dets = np.genfromtxt(det_filename, delimiter=",", dtype=np.str)
      self._dets[seq] = dets

  def load_image(self, img_filename):
    path = tf.string_split([img_filename], ':').values[0]
    return super().load_image(path)

  def load_annotation(self, img, img_filename, annotation_filename):
    class_, bbox = tf.py_func(self._load_annotation, [img_filename], [tf.int64, tf.float32])
    class_.set_shape(())
    bbox.set_shape((4,))
    return {DataKeys.CLASSES: class_, DataKeys.BBOXES_y0x0y1x1: bbox}

  def _load_annotation(self, img_filename):
    img_filename = img_filename.decode("utf-8")
    sp = img_filename.split(":")
    seq = sp[1]
    idx = int(sp[2])
    det = self._dets[seq][idx]
    x = float(det[2])
    y = float(det[3])
    w = float(det[4])
    h = float(det[5])
    bbox_y0x0y1x1 = np.array([y, x, y + h, x + w], np.float32)
    cls = int(sp[3])
    return cls, bbox_y0x0y1x1

  def read_inputfile_lists(self):
    imgs = []
    for seq, dets in self._dets.items():
      for idx, det in enumerate(dets):
        t = int(float(det[0]))
        id_to_t_to_obj = self._gt[seq]
        cls = 0
        x = float(det[2])
        y = float(det[3])
        w = float(det[4])
        h = float(det[5])
        bbox_x0y0x1y1 = np.array([x, y, x + w, y + h], np.float32)
        for t_to_obj in id_to_t_to_obj.values():
          if t in t_to_obj.keys():
            obj = t_to_obj[t]
            iou = calculate_ious(bbox_x0y0x1y1[np.newaxis], obj.bbox_x0y0x1y1[np.newaxis])[0, 0]
            if iou > 0.5:
              cls = 1
              break
        imgs.append(self.data_dir + "/" + seq + "/%06d" % t + ".jpg:" + seq + ":" + str(idx) + ":" + str(cls))
    return imgs,
