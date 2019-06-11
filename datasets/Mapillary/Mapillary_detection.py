from datasets.Loader import register_dataset
from datasets.DetectionDataset import MapillaryLikeDetectionFileListDataset
from datasets.util.Util import username

NAME = "Mapillary_detection"
DATA_LIST_PATH = "datasets/Mapillary/"
DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/mapillary_quarter/"
N_MAX_DETECTIONS = 300
ID_DIVISOR = 256
CLASS_IDS_WITH_INSTANCES = [0, 1, 8, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47,
                            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]
NUM_CLASSES = len(CLASS_IDS_WITH_INSTANCES) + 1


@register_dataset(NAME)
class MapillaryDetectionDataset(MapillaryLikeDetectionFileListDataset):
  def __init__(self, config, subset):
    super().__init__(config, NAME, subset, DEFAULT_PATH, NUM_CLASSES, N_MAX_DETECTIONS, CLASS_IDS_WITH_INSTANCES,
                     ID_DIVISOR)

  def read_inputfile_lists(self):
    data_list = "training.txt" if self.subset == "train" else "validation.txt"
    data_list = DATA_LIST_PATH + "/" + data_list
    img_list = []
    an_list = []
    with open(data_list) as f:
      for l in f:
        im, an, *im_ids_and_sizes = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an
        img_list.append(im)
        an_list.append(an)
    return img_list, an_list
