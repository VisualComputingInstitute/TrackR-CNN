import glob
import os
import numpy as np

from datasets.MOT.MOT_common import MOTDetectionDataset, MOTDataset
from datasets.Loader import register_dataset
from datasets.util.Util import username

# MOT17 is the same sequences as MOT16 with better ground truth annotations
NAME = "PathTrack"
NAME_DETECTION = "PathTrack_detection"
DEFAULT_PATH = "/fastwork/" + username() + "/data/pathtrack_release_v1.0/pathtrack_release"
# PathTrack actually only contains pedestrian annotations; keep the three classes here so that it fits with MOT17
NUM_CLASSES = 3  # background, car, pedestrian
N_MAX_DETECTIONS = 100

# TODO Which split?
PATHTRACK_SEQUENCES = sorted(glob.glob(os.path.join(DEFAULT_PATH, "train", "*")))
PATHTRACK_SEQUENCES = [seq.split("/")[-1] for seq in PATHTRACK_SEQUENCES]
PATHTRACK_SPLITPOINT = int(np.ceil(len(PATHTRACK_SEQUENCES)*0.7))
SEQ_IDS_TRAIN = PATHTRACK_SEQUENCES[:PATHTRACK_SPLITPOINT]
SEQ_IDS_VAL = PATHTRACK_SEQUENCES[PATHTRACK_SPLITPOINT:]


# used for detection on invididual images
@register_dataset(NAME_DETECTION)
class PathTrackDetectionDataset(MOTDetectionDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME, DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL, NUM_CLASSES, ' ')

  def get_data_arrays_for_file(self, img_filename, img_h, img_w):
    return pathtrack_get_data_arrays_for_file(self.gt_data, img_filename, img_h, img_w)


# used for training on chunks of video
@register_dataset(NAME)
class PathTrackDataset(MOTDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, NAME, DEFAULT_PATH, SEQ_IDS_TRAIN, SEQ_IDS_VAL, NUM_CLASSES, ' ')

  def get_data_arrays_for_file(self, img_filename, img_h, img_w):
    return pathtrack_get_data_arrays_for_file(self.gt_data, img_filename, img_h, img_w)


def pathtrack_get_data_arrays_for_file(gt_data, img_filename, img_h, img_w):
  img_filename = img_filename.decode('utf-8')
  seq = img_filename.split("/")[-3]
  img_id = int(img_filename.split("/")[-1][:-4])
  all_anns = gt_data[seq][0]
  anns_for_img = all_anns[all_anns[:, 0] == img_id, :]
  assert (len(anns_for_img) <= N_MAX_DETECTIONS)

  # they need to be padded to N_MAX_DETECTIONS
  bboxes = np.zeros((N_MAX_DETECTIONS, 4), dtype="float32")
  ids = np.zeros(N_MAX_DETECTIONS, dtype="int32")
  classes = np.zeros(N_MAX_DETECTIONS, dtype="int32")
  is_crowd = np.zeros(N_MAX_DETECTIONS, dtype="int32")

  for idx, ann in enumerate(anns_for_img):
    x1 = ann[2]
    y1 = ann[3]
    box_width = ann[4]
    box_height = ann[5]
    x2 = x1 + box_width
    y2 = y1 + box_height
    # clip box
    x1 = np.clip(x1, 0, img_w - 1)
    x2 = np.clip(x2, 0, img_w - 1)
    y1 = np.clip(y1, 0, img_h - 1)
    y2 = np.clip(y2, 0, img_h - 1)

    bboxes[idx] = [y1, x1, y2, x2]
    ids[idx] = ann[1]

    # TODO use the information whether a box was interpolated?
    classes[idx] = 1  # Everything is a pedestrian

  return bboxes, ids, classes, is_crowd
