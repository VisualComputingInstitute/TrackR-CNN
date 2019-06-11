from random import shuffle
import tensorflow as tf

from datasets.Dataset import FileListDataset
from core.Log import log


NUM_CLASSES = 2


class MapillaryLikeInstanceDataset(FileListDataset):
  def __init__(self, config, subset, name, default_path, data_list_path, id_divisor, cat_ids_to_use):
    super().__init__(config, name, subset, default_path, NUM_CLASSES)
    self.validation_set_size = self.config.int("validation_set_size", -1)
    # note: in case of mapillary the min sizes are always based on the sizes in quarter resolution!
    self.min_size = config.int("min_size", 0)
    self._cat_ids_to_use = cat_ids_to_use
    self._data_list_path = data_list_path
    self._id_divisor = id_divisor
    self.name = name

  def read_inputfile_lists(self):
    data_list = "training.txt" if self.subset == "train" else "validation.txt"
    print("{} ({}): using data_dir:".format(self.name, self.subset), self.data_dir, file=log.v5)
    data_list = self._data_list_path + "/" + data_list
    imgs_ans = []
    with open(data_list) as f:
      for l in f:
        im, an, *im_ids_and_sizes = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an
        for id_and_size in im_ids_and_sizes:
          id_ = id_and_size.split(":")[0]
          size_ = int(id_and_size.split(":")[1])
          if self.subset == "train" and size_ < self.min_size:
            continue
          cat_id = int(id_) // self._id_divisor
          if self._cat_ids_to_use is not None and cat_id not in self._cat_ids_to_use:
            continue
          imgs_ans.append((im, an + ":" + id_))
    if self.subset == "train":
      shuffle(imgs_ans)
    elif self.validation_set_size != -1:
      imgs_ans = imgs_ans[:self.validation_set_size]
    imgs = [x[0] for x in imgs_ans]
    ans = [x[1] for x in imgs_ans]
    return imgs, ans

  def load_annotation(self, img, img_filename, annotation_filename):
    annotation_filename_without_id = tf.string_split([annotation_filename], ':').values[0]
    ann_data = tf.read_file(annotation_filename_without_id)
    ann = tf.image.decode_png(ann_data, dtype=tf.uint16, channels=1)
    ann.set_shape(img.get_shape().as_list()[:-1] + [1])
    ann = self.postproc_annotation(annotation_filename, ann)
    return ann

  def postproc_annotation(self, ann_filename, ann):
    id_str = tf.string_split([ann_filename], ':').values[1]
    id_ = tf.string_to_number(id_str, out_type=tf.int32)
    ann_postproc = tf.cast(tf.equal(tf.cast(ann, tf.int32), id_), tf.uint8)
    return ann_postproc
