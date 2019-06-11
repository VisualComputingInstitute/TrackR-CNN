import glob
import tensorflow as tf

from datasets.DetectionDataset import MapillaryLikeDetectionFileListDataset
from datasets.KITTI.segtrack.KITTI_MOTS_info import SEQ_IDS_TRAIN, SEQ_IDS_VAL
from datasets.Loader import register_dataset
from datasets.util.Util import username
from datasets import DataKeys
from shutil import copytree
from random import randint
from core.Log import log
import os

NAME = "KITTI_segtrack"
NAME_DETECTION = "KITTI_segtrack_detection"
DEFAULT_PATH = "/globalwork/" + username() + "/data/KITTI_MOTS/train/"
ID_DIVISOR = 1000
CLASS_IDS_WITH_INSTANCES = [1, 2]
CROWD_ID = 10
NUM_CLASSES = 3  # background, car, pedestrian
N_MAX_DETECTIONS = 100


# used for detection on individual images
@register_dataset(NAME_DETECTION)
class KittiSegtrackDetectionDataset(MapillaryLikeDetectionFileListDataset):
  def __init__(self, config, subset, name=NAME, default_path=DEFAULT_PATH):
    self.seq_ids_train = SEQ_IDS_TRAIN
    self.seq_ids_val = SEQ_IDS_VAL
    self.imgs_are_pngs = config.bool("imgs_are_pngs", True)
    t = config.string_list("seq_ids_train", [])
    if t:
      self.seq_ids_train = t
    v = config.string_list("seq_ids_val", [])
    if v:
      self.seq_ids_val = v
    super().__init__(config, name, subset, default_path, NUM_CLASSES, N_MAX_DETECTIONS, CLASS_IDS_WITH_INSTANCES,
                     ID_DIVISOR, crowd_id=CROWD_ID)
    self.copy_dataset_to_tmp = config.bool("copy_dataset_to_tmp", False)
    if self.copy_dataset_to_tmp:
      print("Copying dataset to $TMP!", file=log.v1)
      new_path = "$TMP/" + str(randint(1, 100000))
      new_path = os.path.expandvars(new_path)
      os.makedirs(new_path)
      print("Copying images...", file=log.v1)
      copytree(self.data_dir + "/images", new_path + "/images")
      print("Copying instances...", file=log.v1)
      copytree(self.data_dir + "/instances", new_path + "/instances")
      self.data_dir = new_path

  def read_inputfile_lists(self):
    seq_ids = self.seq_ids_train if self.subset == "train" else self.seq_ids_val
    anns = []
    for seq_id in seq_ids:
      anns += sorted(glob.glob(self.data_dir + "/instances/" + seq_id + "/*.png"))

    imgs = [x.replace("/instances/", "/images/") for x in anns]
    if not self.imgs_are_pngs:
      imgs = [x.replace(".png", ".jpg") for x in imgs]
    return imgs, anns


# used for training on chunks of video
@register_dataset(NAME)
class KittiSegtrackDataset(KittiSegtrackDetectionDataset):
  def __init__(self, config, subset, name=NAME, default_path=DEFAULT_PATH):
    # batch size here is the number of time steps considered in a chunk
    # TODO: what do we do at test time?
    self._batch_size = config.int("batch_size")
    assert self._batch_size > 1, "use KittiSegtrackDetectionDataset for single image training"
    super().__init__(config, subset, name, default_path)

  def read_inputfile_lists(self):
    seq_ids = self.seq_ids_train if self.subset == "train" else self.seq_ids_val
    anns = []
    for seq_id in seq_ids:
      anns_vid = sorted(glob.glob(self.data_dir + "/instances/" + seq_id + "/*.png"))
      starting_points = anns_vid[:-(self._batch_size - 1)]
      anns += starting_points

    imgs = [x.replace("/instances/", "/images/") for x in anns]
    return imgs, anns

  def _batch(self, tfdata, batch_size):
    return tfdata

  def load_example(self, input_filenames):
    examples = []
    for delta_t in range(self._batch_size):
      input_filenames_t = [successor_frame_filename(fn, delta_t) for fn in input_filenames]
      raw_example = self.load_raw_example(*input_filenames_t)
      examples.append(raw_example)

    # cf process_raw_example
    # here we need to do it jointly to synchronize the augmentation (e.g. flipping)
    examples = [self.postproc_example_initial(example) for example in examples]
    examples = self.jointly_augment_examples_before_resize(examples)
    examples = [self.postproc_example_before_resize(example) for example in examples]
    examples = [self.resize_example(example) for example in examples]
    examples = self.jointly_augment_examples_after_resize(examples)
    examples = [self.postproc_example_before_assembly(example) for example in examples]
    examples = [self.assemble_example(example) for example in examples]

    # stack everything together
    examples_stacked = {}
    for key in examples[0].keys():
      if key == DataKeys.SKIP_EXAMPLE:
        stacked = tf.reduce_any([example[key] for example in examples])
      else:
        stacked = tf.stack([example[key] for example in examples], axis=0)
      examples_stacked[key] = stacked
    return examples_stacked

  def n_examples_per_epoch(self):
    n_examples = super().n_examples_per_epoch()
    if n_examples == len(self.inputfile_lists[0]):
      return n_examples * self._batch_size


def successor_frame_filename(filename, offset):
  if offset == 0:
    return filename
  else:
    return tf.py_func(successor_frame_filename_np, [filename, offset], tf.string)


def successor_frame_filename_np(filename, offset):
  filename = filename.decode("utf-8")
  sp = filename.split("/")
  png = ".png" in sp[-1]
  t = int(sp[-1].replace(".png", "").replace(".jpg", ""))
  if png:
    filename = "/".join(sp[:-1]) + "/%06d" % (t + offset) + ".png"
  else:
    filename = "/".join(sp[:-1]) + "/%06d" % (t + offset) + ".jpg"
  return filename.encode("utf-8")
