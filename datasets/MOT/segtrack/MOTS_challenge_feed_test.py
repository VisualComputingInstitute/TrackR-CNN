import glob
from datasets.Loader import register_dataset
from datasets.KITTI.segtrack.KITTI_segtrack_feed import KittiSegtrackLikeFeedDataset

NAME = "MOTS_challenge_feed_test"
DEFAULT_PATH = "/globalwork/voigtlaender/data/MOTS_challenge/test/"

SEQ_IDS_TRAIN = []
SEQ_IDS_VAL = ["%04d" % idx for idx in [1, 6, 7, 12]]
TIMESTEPS_PER_SEQ = {"0001": 450, "0006": 1194, "0007": 500, "0012": 900}

@register_dataset(NAME)
class MOTSSegtrackFeedDataset(KittiSegtrackLikeFeedDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, "MOTS_challenge_test", DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL, False)
    self.time_starts_at_1 = True

  def get_filenames_for_video_idx(self, idx):
    return sorted(glob.glob(self.data_dir + "/images/" + self._video_tags[idx] + "/*.jpg"))
