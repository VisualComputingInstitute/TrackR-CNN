import glob
import numpy as np
from collections import namedtuple, OrderedDict


def load_tracking_gt_KITTI_format(gt_path, filter_to_cars=True, start_track_ids_from_1=True):
  tracking_gt = {}
  tracking_gt_files = glob.glob(gt_path + "/*.txt")
  for tracking_gt_file in tracking_gt_files:
    seq = tracking_gt_file.split("/")[-1].replace(".txt", "")
    print("loading", tracking_gt_file)
    gt = np.genfromtxt(tracking_gt_file, dtype=np.str)
    if filter_to_cars:
      gt = gt[gt[:, 2] != "Cyclist"]
      gt = gt[gt[:, 2] != "Pedestrian"]
      gt = gt[gt[:, 2] != "Person"]
    gt = gt[gt[:, 2] != "DontCare"]
    if start_track_ids_from_1:
      # increase all track ids by 1 so we start at 1 instead of 0
      gt[:, 1] = (gt[:, 1].astype(np.int32) + 1).astype(np.str)

    tracking_gt[seq] = gt
  return tracking_gt


TrackingGtElement = namedtuple("TrackingGtElement", "time class_ id_ bbox_x0y0x1y1")


def load_tracking_gt_KITTI(gt_path, filter_to_cars_and_pedestrians, cars_only=False, pedestrians_only=False):
  gt = load_tracking_gt_KITTI_format(gt_path, filter_to_cars=False, start_track_ids_from_1=False)
  if filter_to_cars_and_pedestrians:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] != "Cyclist"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Van"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Person"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Tram"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Truck"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Misc"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  if cars_only:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] == "Car"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  if pedestrians_only:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] == "Pedestrian"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  # convert to TrackingGTElements which are much easier to use
  # and store it in nested dicts: seq -> id -> time -> TrackingGtElement
  nice_gt = {}
  for seq, seq_gt in gt.items():
    id_to_time_to_elem = OrderedDict()
    for gt_elem in seq_gt:
      t = int(gt_elem[0])
      id_ = int(gt_elem[1])
      class_ = gt_elem[2]
      bbox_x0y0x1y1 = gt_elem[6:10].astype("float")
      elem = TrackingGtElement(time=t, class_=class_, id_=id_, bbox_x0y0x1y1=bbox_x0y0x1y1)
      time_to_elem = id_to_time_to_elem.setdefault(id_, OrderedDict())
      time_to_elem[t] = elem
    nice_gt[seq] = id_to_time_to_elem
  return nice_gt


TrackingDetElement = namedtuple("TrackingDetElement", "time class_ id_ bbox_x0y0x1y1 score")


def load_tracking_det_KITTI(gt_path, filter_to_cars_and_pedestrians, cars_only=False, pedestrians_only=False):
  gt = load_tracking_gt_KITTI_format(gt_path, filter_to_cars=False, start_track_ids_from_1=False)
  if filter_to_cars_and_pedestrians:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] != "Cyclist"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Van"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Person"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Tram"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Truck"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Misc"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  if cars_only:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] == "Car"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  if pedestrians_only:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] == "Pedestrian"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  # convert to TrackingGTElements which are much easier to use
  # and store it in nested dicts: seq -> id -> time -> TrackingGtElement
  nice_gt = {}
  for seq, seq_gt in gt.items():
    id_to_time_to_elem = OrderedDict()
    for gt_elem in seq_gt:
      t = int(gt_elem[0])
      id_ = int(gt_elem[1])
      class_ = gt_elem[2]
      bbox_x0y0x1y1 = gt_elem[6:10].astype("float")
      score = gt_elem[17].astype("float")
      elem = TrackingDetElement(time=t, class_=class_, id_=id_, bbox_x0y0x1y1=bbox_x0y0x1y1, score=score)
      time_to_elem = id_to_time_to_elem.setdefault(id_, OrderedDict())
      time_to_elem[t] = elem
    nice_gt[seq] = id_to_time_to_elem
  return nice_gt


def load_tracking_gt_mot(gt_path, filter_invisible=False, tracking_gt_files=None):
  tracking_gt = {}
  if tracking_gt_files is None:
    tracking_gt_files = glob.glob(gt_path + "*.txt")
  for tracking_gt_file in tracking_gt_files:
    seq = tracking_gt_file.split("/")[-1].replace(".txt", "")
    ## for some reason this causes trouble for some lines...
    gt = np.genfromtxt(tracking_gt_file, delimiter=",", dtype=np.str)
    tracking_gt[seq] = gt

  # convert to TrackingGTElements which are much easier to use
  # and store it in nested dicts: seq -> id -> time -> TrackingGtElement
  nice_gt = {}
  for seq, seq_gt in tracking_gt.items():
    id_to_time_to_elem = OrderedDict()
    for gt_elem in seq_gt:
      t = int(gt_elem[0])
      id_ = int(gt_elem[1])
      class_ = int(gt_elem[7])
      # Only use pedestrians for now
      if class_ != 1:
        continue
      bbox_x0y0wh = gt_elem[2:6].astype("float")
      x0 = bbox_x0y0wh[0]
      y0 = bbox_x0y0wh[1]
      w = bbox_x0y0wh[2]
      h = bbox_x0y0wh[3]
      x1 = x0 + w
      y1 = y0 + h
      bbox_x0y0x1y1 = np.array([x0, y0, x1, y1])
      #visibility = float(gt_elem[8])
      # Ignore totally occluded boxes
      #if visibility == 0 and filter_invisible:
      #  continue
      elem = TrackingGtElement(time=t, class_=class_, id_=id_, bbox_x0y0x1y1=bbox_x0y0x1y1)
      time_to_elem = id_to_time_to_elem.setdefault(id_, OrderedDict())
      time_to_elem[t] = elem
    nice_gt[seq] = id_to_time_to_elem
  return nice_gt
