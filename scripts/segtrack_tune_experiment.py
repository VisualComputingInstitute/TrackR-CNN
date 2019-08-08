import os
import sys
import subprocess
import random
from functools import partial
from collections import namedtuple
#from datasets.KITTI.segtrack.KITTI_MOTS_info import TIMESTEPS_PER_SEQ #SEQ_IDS_TRAIN, SEQ_IDS_VAL
from forwarding.tracking.TrackingForwarder_util import make_disjoint, export_tracking_result_in_kitti_format, \
  import_detections_for_sequence, load_optical_flow
from forwarding.tracking.Util_tracking import track_single_sequence
from multiprocessing import Pool, cpu_count

Scores = namedtuple("Scores", ["sMOTSA_car", "sMOTSA_ped", "MOTSA_car", "MOTSA_ped", "MOTSP_car", "MOTSP_ped",
                               "IDS_car", "IDS_ped"])


def read_result(content):
  lines = content.split("\n")
  cars_index = -1
  ped_index = -1
  for line_index in range(len(lines)):
    if "Evaluate class: Cars" in lines[line_index]:
      cars_index = line_index
    if "Evaluate class: Pedestrians" in lines[line_index]:
      ped_index = line_index
  results_cars = lines[cars_index+2].split()
  results_ped = lines[ped_index+2].split()

  score = Scores(float(results_cars[1]), float(results_ped[1]), float(results_cars[2]), float(results_ped[2]), float(results_cars[3]),
                 float(results_ped[3]), int(results_cars[17]), int(results_ped[17]))
  print(score)
  return score


def process_sequence(seq, detections_path, add_masks, tracker_options, optical_flow_path, temp_out, n_timesteps=None,
                     start_time_at_1=False):
  assert n_timesteps is not None
  #n_timesteps = TIMESTEPS_PER_SEQ[seq]
  det_boxes, det_scores, reid_features, det_classes, det_masks = \
    import_detections_for_sequence(seq, n_timesteps, detections_path, "", 0, add_masks)

  if tracker_options["mask_iou_weight_car"] > 0.0 or \
      tracker_options["mask_iou_weight_pedestrian"] > 0.0 or \
      tracker_options["bbox_iou_weight_car"] > 0.0 or \
      tracker_options["bbox_iou_weight_pedestrian"] > 0.0:
    optical_flow = load_optical_flow(seq, optical_flow_path)
  else:
    optical_flow = None

  hyp_tracks = track_single_sequence(tracker_options, det_boxes, det_scores, reid_features, det_classes,
                                     det_masks, optical_flow=optical_flow)
  hyp_tracks = make_disjoint(hyp_tracks, "score")
  export_tracking_result_in_kitti_format(seq, hyp_tracks, add_masks, "", temp_out, start_time_at_1=start_time_at_1)


def run_experiment(tracker_options, detections_path, gt_path, optical_flow_path, temp_path, eval_path, seqmap,
                   add_masks=True):
  temp_out = temp_path + "/" + str(random.randint(0, 8987521254))
  # derive sequences from seqmap
  sequences_with_timesteps = []
  start_time_at_1 = False
  with open(os.path.join(eval_path, seqmap)) as f:
      for l in f:
          sp = l.strip().split()
          start = int(sp[2])
          # assume start times can only be 0 or 1 and that they are the same for all sequences
          assert start in (0, 1), ("unexpected start time", start)
          if start == 1:
              start_time_at_1 = True
          timesteps = int(sp[3])
          if start == 0:
              timesteps += 1
          sequences_with_timesteps.append((sp[0], timesteps))
  print(sequences_with_timesteps, start_time_at_1)
  proc_sec_partial = partial(process_sequence, detections_path=detections_path, add_masks=add_masks,
                             tracker_options=tracker_options, optical_flow_path=optical_flow_path, temp_out=temp_out,
                             start_time_at_1=start_time_at_1)
  for seq, n_timesteps in sequences_with_timesteps:
      proc_sec_partial(seq, n_timesteps=n_timesteps)

  p = subprocess.run(["python3", "eval.py", temp_out, gt_path, seqmap],
                     stdout=subprocess.PIPE, cwd=eval_path)
  results = p.stdout.decode("utf-8")
  score = read_result(results)
  return score


def tune_model(detections_path, gt_path, output_path, optical_flow_path, temp_path, eval_path,
               num_experiments, tune_what="bbox_center", add_masks=True, seq_map_train="train.seqmap", seq_map_val="val.seqmap"):
  tracker_options = {"tracker": "hungarian", "reid_comp": "euclidean",
                     "detection_confidence_threshold_car": 0.55,
                     "detection_confidence_threshold_pedestrian": 0.95,
                     "reid_weight_car": 0.0,
                     "reid_weight_pedestrian": 0.0,
                     "mask_iou_weight_car": 0.0,
                     "mask_iou_weight_pedestrian": 0.0,
                     "bbox_center_weight_car": 0.0,
                     "bbox_center_weight_pedestrian": 0.0,
                     "bbox_iou_weight_car": 0.0,
                     "bbox_iou_weight_pedestrian": 0.0,
                     "association_threshold_car": 0.3,
                     "association_threshold_pedestrian": 0.3,
                     "keep_alive_car": 0,
                     "keep_alive_pedestrian": 0,
                     "reid_euclidean_offset_car": 5.0,
                     "reid_euclidean_scale_car": 1.0,
                     "reid_euclidean_offset_pedestrian": 5.0,
                     "reid_euclidean_scale_pedestrian": 1.0,
                     "new_reid_threshold_car": 2.0,
                     "new_reid_threshold_pedestrian": 2.0,
                     "box_offset": 50.0,
                     "box_scale": 0.02,
                     "new_reid": False}

  expr_tracker_opts = []
  for i in range(num_experiments):
    exp_options = tracker_options.copy()
    if tune_what == "bbox_center":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["bbox_center_weight_car"] = 1.0
      exp_options["bbox_center_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.99)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.99)
      exp_options["box_offset"] = random.uniform(30.0, 80.0)
      exp_options["box_scale"] = random.uniform(0.01, 0.1)
    elif tune_what == "reid":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["reid_weight_car"] = 1.0
      exp_options["reid_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.99)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.99)
      exp_options["reid_euclidean_offset_car"] = random.uniform(5.0, 10.0)
      exp_options["reid_euclidean_scale_car"] = random.uniform(0.7, 1.7)
      exp_options["reid_euclidean_offset_pedestrian"] = exp_options["reid_euclidean_offset_car"]
      exp_options["reid_euclidean_scale_pedestrian"] = exp_options["reid_euclidean_scale_car"]
      exp_options["keep_alive_car"] = random.randint(0, 4)
      exp_options["keep_alive_pedestrian"] = random.randint(6, 10)
    elif tune_what == "reid_norm_euc":
      exp_options["reid_comp"] = "normalized_euclidean"
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["reid_weight_car"] = 1.0
      exp_options["reid_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(-0.99, 0.99)
      exp_options["association_threshold_pedestrian"] = random.uniform(-0.99, 0.99)
    elif tune_what == "new_reid":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["mask_iou_weight_car"] = 1.0
      exp_options["mask_iou_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.2)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.2)
      exp_options["new_reid"] = True
      exp_options["new_reid_threshold_car"] = random.uniform(0.7, 1.7)
      exp_options["new_reid_threshold_pedestrian"] = random.uniform(0.7, 1.7)
    elif tune_what == "new_reid_box":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["bbox_iou_weight_car"] = 1.0
      exp_options["bbox_iou_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.2)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.2)
      exp_options["new_reid"] = True
      exp_options["new_reid_threshold_car"] = random.uniform(0.7, 1.7)
      exp_options["new_reid_threshold_pedestrian"] = random.uniform(0.7, 1.7)
    elif tune_what == "mask":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["mask_iou_weight_car"] = 1.0
      exp_options["mask_iou_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.2)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.2)
    elif tune_what == "bbox_iou":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      exp_options["bbox_iou_weight_car"] = 1.0
      exp_options["bbox_iou_weight_pedestrian"] = 1.0
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.2)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.2)
    elif tune_what == "mask_reid":
      exp_options["detection_confidence_threshold_car"] = random.uniform(0.5, 0.85)
      exp_options["detection_confidence_threshold_pedestrian"] = random.uniform(0.8, 0.99)
      l = random.uniform(0.0, 1.0)
      exp_options["mask_iou_weight_car"] = l
      exp_options["mask_iou_weight_pedestrian"] = l
      exp_options["reid_weight_car"] = 1.0 - l
      exp_options["reid_weight_pedestrian"] = 1.0 - l
      exp_options["association_threshold_car"] = random.uniform(0.0, 0.99)
      exp_options["association_threshold_pedestrian"] = random.uniform(0.0, 0.99)
      exp_options["reid_euclidean_offset_car"] = random.uniform(5.0, 10.0)
      exp_options["reid_euclidean_scale_car"] = random.uniform(0.7, 1.7)
      exp_options["reid_euclidean_offset_pedestrian"] = exp_options["reid_euclidean_offset_car"]
      exp_options["reid_euclidean_scale_pedestrian"] = exp_options["reid_euclidean_scale_car"]
    else:
      assert False, tune_what
    expr_tracker_opts.append(exp_options)

  run_exp_partial = partial(run_experiment, detections_path=detections_path, gt_path=gt_path, add_masks=add_masks,
                            optical_flow_path=optical_flow_path, temp_path=temp_path, eval_path=eval_path)
  run_exp_partialT = partial(run_exp_partial, seqmap=seq_map_train)
  run_exp_partialV = partial(run_exp_partial, seqmap=seq_map_val)
  pool = Pool(cpu_count())
  expr_scores = pool.map(run_exp_partialT, expr_tracker_opts)
  #expr_scores = list(map(run_exp_partialT, expr_tracker_opts))

  with open(output_path, "w") as f:
    for settings, scores in zip(expr_tracker_opts, expr_scores):
      print(settings, scores, file=f)

  def find_best(score_to_tune="sMOTSA_car"):
    best_index = -1
    best_score = -float("Inf")
    for index, score in enumerate(expr_scores):
      if getattr(score, score_to_tune) > best_score:
        best_index = index
        best_score = getattr(score, score_to_tune)
    return best_index

  best_car = find_best("sMOTSA_car")
  print("Best CAR sMOTSA train", getattr(expr_scores[best_car], "sMOTSA_car"), "Settings:", {key: val for (key, val) in expr_tracker_opts[best_car].items() if not "ped" in key})
  best_car_val_scores = run_exp_partialV(expr_tracker_opts[best_car])
  print("Val scores", "sMOTSA", getattr(best_car_val_scores, "sMOTSA_car"), "MOTSA", getattr(best_car_val_scores, "MOTSA_car"),
        "MOTSP", getattr(best_car_val_scores, "MOTSP_car"), "IDS", getattr(best_car_val_scores, "IDS_car"))
  print("---")
  best_ped = find_best("sMOTSA_ped")
  print("Best PED sMOTSA train", getattr(expr_scores[best_ped], "sMOTSA_ped"), "Settings:", {key: val for (key, val) in expr_tracker_opts[best_ped].items() if not "car" in key})
  best_ped_val_scores = run_exp_partialV(expr_tracker_opts[best_ped])
  print("Val scores", "sMOTSA", getattr(best_ped_val_scores, "sMOTSA_ped"), "MOTSA", getattr(best_ped_val_scores, "MOTSA_ped"),
        "MOTSP", getattr(best_ped_val_scores, "MOTSP_ped"), "IDS", getattr(best_ped_val_scores, "IDS_ped"))


def main():
  assert len(sys.argv) in (9, 11), "usage: <filename>.py <detections> <gt> <optflow> <output> <tmp> <eval> <tune_what> <num_exp> [seq_map_train seq_map_val]"

  detections_path = sys.argv[1]
  gt_path = sys.argv[2]
  optical_flow_path = sys.argv[3]
  output_path = sys.argv[4]
  temp_path = sys.argv[5]
  eval_path = sys.argv[6]
  tune_what = sys.argv[7]
  num_exp = int(sys.argv[8])
  if len(sys.argv) == 11:
      seq_map_train = sys.argv[9]
      seq_map_val = sys.argv[10]
  else:
      seq_map_train = "train.seqmap"
      seq_map_val = "val.seqmap"

  tune_model(detections_path, gt_path, output_path, optical_flow_path, temp_path, eval_path, num_exp, tune_what, seq_map_train=seq_map_train, seq_map_val=seq_map_val)


if __name__ == '__main__':
  main()
