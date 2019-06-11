import glob

from datasets.KITTI.segtrack.KITTI_segtrack import KittiSegtrackDataset
from datasets.Loader import register_dataset

NAME = "MOTS_challenge"
DEFAULT_PATH = "/globalwork/voigtlaender/data/MOTS_challenge/train/"
SEQ_IDS_TRAIN = ["%04d" % idx for idx in [2, 5, 9, 11]]
SEQ_IDS_VAL = ["%04d" % idx for idx in [2, 5, 9, 11]]
TIMESTEPS_PER_SEQ = {"0002": 600, "0005": 837, "0009": 525, "0011": 900}


@register_dataset(NAME)
class MotsChallengeDataset(KittiSegtrackDataset):
  def __init__(self, config, subset):
    #TODO: num classes, class_ids_with instances
    super().__init__(config, subset, name=NAME, default_path=DEFAULT_PATH)
    self.seq_ids_train = SEQ_IDS_TRAIN
    self.seq_ids_val = SEQ_IDS_VAL
    t = config.string_list("seq_ids_train", [])
    if t:
      self.seq_ids_train = t
    v = config.string_list("seq_ids_val", [])
    if v:
      self.seq_ids_val = v


  def read_inputfile_lists(self):
    #seq_ids = SEQ_IDS_TRAIN if self.subset == "train" else SEQ_IDS_VAL
    seq_ids = self.seq_ids_train if self.subset == "train" else self.seq_ids_val
    anns = []
    for seq_id in seq_ids:
      anns_vid = sorted(glob.glob(self.data_dir + "/instances/" + seq_id + "/*.png"))
      starting_points = anns_vid[:-(self._batch_size - 1)]
      anns += starting_points

    imgs = [x.replace("/instances/", "/images/").replace(".png", ".jpg") for x in anns]
    return imgs, anns

