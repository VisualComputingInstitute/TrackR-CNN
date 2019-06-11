import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool
from abc import abstractmethod

from datasets import DataKeys
from datasets.Loader import register_dataset
from datasets.FeedDataset import FeedDataset
from datasets.KITTI.segtrack.KITTI_segtrack import NUM_CLASSES, DEFAULT_PATH
from datasets.KITTI.segtrack.KITTI_MOTS_info import SEQ_IDS_TRAIN, SEQ_IDS_VAL
from datasets.util.Detection import init_anchors
from core.Log import log

NAME = "KITTI_segtrack_feed"
DATA_KEYS_TO_USE = [DataKeys.IMAGES, DataKeys.IMAGE_FILENAMES]


class KittiSegtrackLikeFeedDataset(FeedDataset):
  def __init__(self, config, subset, dataset_name, default_path, seq_ids_train, seq_ids_val, preload_images):
    super().__init__(config, subset, data_keys_to_use=DATA_KEYS_TO_USE, num_classes=NUM_CLASSES)
    self.time_starts_at_1 = False
    self.data_dir = config.string(dataset_name + "_data_dir", default_path)
    video_tags_config = config.string_list("video_tags_to_load", [])
    if len(video_tags_config) > 0:
      self._video_tags = video_tags_config
    else:
      if self.subset == "train":
        self._video_tags = seq_ids_train
      else:
        self._video_tags = seq_ids_val
    self._video_idx = None
    self._imgs = None
    self._curr_time = None
    self._filenames = None
    self._preload_images = preload_images
    init_anchors(config)
    # This is somewhat hacky
    self._num_context_frames = 0
    if self.config.bool("offset_matching", False):
      self._num_context_frames = self.config.int("num_space_time_targets", 1)
      print("Supplying", self._num_context_frames, "context frames so actual batch size will decrease")

  def set_video_idx(self, idx):
    self._curr_time = 0
    if self._video_idx == idx:
      return
    self._video_idx = idx
    self._filenames = self.get_filenames_for_video_idx(idx)
    if self.config.bool("short_videos_for_testing", False):
      print("Warning, shortening video to 2 frames for testing", file=log.v1)
      self._filenames = self._filenames[:2]
    if self._preload_images:
      print("loading images for seq", self.get_video_tag(), file=log.v5)
      with Pool(8) as pool:
        self._imgs = pool.map(_load_img, self._filenames)
    print("done", file=log.v5)

  @abstractmethod
  def get_filenames_for_video_idx(self, idx):
    raise NotImplementedError

  def n_videos(self):
    return len(self._video_tags)

  def n_examples_per_epoch(self):
    assert self._video_idx is not None
    return len(self._filenames)

  def get_video_tag(self):
    assert self._video_idx is not None
    return self._video_tags[self._video_idx]

  def get_feed_dict_for_next_step(self):
    assert self._video_idx is not None
    feed_dict = {}
    for idx in range(self._batch_size):
      if self._curr_time > 0:
        time_idx = self._curr_time - self._num_context_frames + idx
      else:
        time_idx = self._curr_time + idx
      # On the last batch, repeat the final image to fill up batch
      if time_idx >= len(self._filenames):
        time_idx = len(self._filenames) - 1
      if self._preload_images:
        feed_dict[self._placeholders[DataKeys.IMAGES][idx]] = self._imgs[time_idx]
      else:
        feed_dict[self._placeholders[DataKeys.IMAGES][idx]] = _load_img(self._filenames[time_idx])
      feed_dict[self._placeholders[DataKeys.IMAGE_FILENAMES][idx]] = self._filenames[time_idx]
    self._curr_time += self._batch_size - self._num_context_frames
    return feed_dict


@register_dataset(NAME)
class KittiSegtrackFeedDataset(KittiSegtrackLikeFeedDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, "KITTI_segtrack", DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL, False)

  def get_filenames_for_video_idx(self, idx):
    return sorted(glob.glob(self.data_dir + "/images/" + self._video_tags[idx] + "/*.png"))


def _load_img(filename):
  return np.array(Image.open(filename), dtype="float32") / 255
