import glob
import os

from datasets.Loader import register_dataset
from datasets.KITTI.segtrack.KITTI_segtrack_feed import KittiSegtrackLikeFeedDataset


class MOTFeedDataset(KittiSegtrackLikeFeedDataset):
  def __init__(self, config, subset, name, default_path, seq_ids_train, seq_ids_val):
    super().__init__(config, subset, name, default_path, seq_ids_train, seq_ids_val, False)

  def get_filenames_for_video_idx(self, idx):
    path = os.path.join(self.data_dir, "train", self._video_tags[idx], "img1", "*.jpg")
    return sorted(glob.glob(path))


@register_dataset("MOT17_feed")
class MOT17FeedDataset(MOTFeedDataset):
  def __init__(self, config, subset):
    from datasets.MOT.MOT17 import DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL
    super().__init__(config, subset, "MOT17", DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL)


@register_dataset("2DMOT2015_feed")
class MOT15FeedDataset(MOTFeedDataset):
  def __init__(self, config, subset):
    from datasets.MOT.MOT15 import DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL
    super().__init__(config, subset, "MOT17", DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL)

@register_dataset("PathTrack_feed")
class PathTrackFeedDataset(MOTFeedDataset):
  def __init__(self, config, subset):
    from datasets.MOT.PathTrack import DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL
    super().__init__(config, subset, "PathTrack", DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL)