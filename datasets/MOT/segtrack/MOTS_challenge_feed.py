import glob
from datasets.Loader import register_dataset
from datasets.KITTI.segtrack.KITTI_segtrack_feed import KittiSegtrackLikeFeedDataset

NAME = "MOTS_challenge_feed"
DEFAULT_PATH = "/globalwork/voigtlaender/data/MOTS_challenge/train/"

SEQ_IDS_TRAIN = []
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 5, 9, 11]]


@register_dataset(NAME)
class MOTSSegtrackFeedDataset(KittiSegtrackLikeFeedDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, "MOTS_challenge", DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL, False)
    self.time_starts_at_1 = True

  def get_filenames_for_video_idx(self, idx):
    return sorted(glob.glob(self.data_dir + "/images/" + self._video_tags[idx] + "/*.jpg"))
