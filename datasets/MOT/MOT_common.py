import glob
import os
import tensorflow as tf
import numpy as np

from datasets.DetectionDataset import DetectionFileListDataset
from datasets import DataKeys

NUM_CLASSES = 3  # background, car, pedestrian
N_MAX_DETECTIONS = 100


# used for detection on invididual images
class MOTDetectionDataset(DetectionFileListDataset):
  def __init__(self, config, subset, name, default_path, seq_ids_train, seq_ids_val,
               num_classes=NUM_CLASSES, gt_delimiter=','):
    super().__init__(config, name, subset, default_path, num_classes)
    self.seq_ids = seq_ids_train if self.subset == "train" else seq_ids_val
    self.gt_data = {seq: [] for seq in self.seq_ids}
    self.visibility_threshold = config.float("visibility_threshold", 0.5)  # Boxes with visibility < this are ignored

    parent_folder = os.path.join(self.data_dir, "train")
    for seq in self.seq_ids:
      gt_filename = os.path.join(parent_folder, seq, "gt", "gt.txt")
      self.gt_data[seq].append(np.genfromtxt(gt_filename, delimiter=gt_delimiter))

    self.classes_to_cat = [0, 1, 3]
    self.cat_to_class = {v: i for i, v in enumerate(self.classes_to_cat)}

  def read_inputfile_lists(self):
    parent_folder = os.path.join(self.data_dir, "train")
    imgs = []
    for seq in self.seq_ids:
      imgs += sorted(glob.glob(os.path.join(parent_folder, seq, "img1", "/*.jpg")))
    return (imgs, )

  def load_annotation(self, img, img_filename, annotation_filename):
    img_shape = tf.shape(img)
    bboxes, ids, classes, is_crowd = tf.py_func(self.get_data_arrays_for_file,
                                                      [img_filename, img_shape[0], img_shape[1]],
                                                      [tf.float32, tf.int32, tf.int32, tf.int32],
                                                      name="get_data_arrays_for_file")
    bboxes.set_shape((N_MAX_DETECTIONS, 4))
    ids.set_shape((N_MAX_DETECTIONS,))
    classes.set_shape((N_MAX_DETECTIONS,))
    is_crowd.set_shape((N_MAX_DETECTIONS,))

    return_dict = {}
    return_dict[DataKeys.BBOXES_y0x0y1x1] = bboxes
    return_dict[DataKeys.CLASSES] = classes
    return_dict[DataKeys.IDS] = ids
    return_dict[DataKeys.IS_CROWD] = is_crowd

    return return_dict


# used for training on chunks of video
class MOTDataset(MOTDetectionDataset):
  def __init__(self, config, subset, name, default_path, seq_ids_train, seq_ids_val,
               num_classes=NUM_CLASSES, gt_delimiter=','):
    super().__init__(config, subset, name, default_path, seq_ids_train, seq_ids_val, num_classes, gt_delimiter)
    # batch size here is the number of time steps considered in a chunk
    # TODO: what do we do at test time?
    self._batch_size = self.config.int("batch_size")
    assert self._batch_size > 1, "use MOTDetectionDataset for single image training"

  def read_inputfile_lists(self):
    parent_folder = os.path.join(self.data_dir, "train")
    imgs = []
    for seq in self.seq_ids:
      all_imgs = sorted(glob.glob(os.path.join(parent_folder, seq, "img1", "*.jpg")))
      starting_points = all_imgs[:-(self._batch_size - 1)]
      imgs += starting_points
    return (imgs, )

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
  t = int(sp[-1].replace(".jpg", ""))
  filename = "/".join(sp[:-1]) + "/%06d" % (t + offset) + ".jpg"
  return filename.encode("utf-8")
