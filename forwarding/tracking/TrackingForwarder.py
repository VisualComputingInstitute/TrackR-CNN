import numpy as np
import os
from time import perf_counter

from pycocotools import mask as cocomask
from scipy.misc import imresize
import pycocotools.mask as cocomask
import subprocess

from core import Extractions
from datasets import DataKeys
from core.Log import log
from forwarding.RecurrentDetectionForwarder import RecurrentDetectionForwarder, DETECTION_EXTRACTION_KEYS, \
  visualize_detections
from forwarding.tracking.TrackingForwarder_util import make_disjoint, export_tracking_result_in_kitti_format, export_detections_for_sequence, \
  import_detections_for_sequence, load_optical_flow
from forwarding.tracking.Util_tracking import track_single_sequence
from forwarding.Util import save_embeddings
from datasets.util.Util import username


class TrackingForwarder(RecurrentDetectionForwarder):
  def __init__(self, engine):
    super().__init__(engine, extraction_keys=DETECTION_EXTRACTION_KEYS + (Extractions.REID_FEATURES,))
    self.add_masks = self.config.bool("add_masks", True)
    self.export_embeddings = self.config.bool("export_embeddings", False)

    tracker_reid_comp = self.config.string("tracker_reid_comp", "sigmoid_dot")
    tracker = self.config.string("tracker", "greedy")
    detection_confidence_threshold_car = self.config.float("detection_confidence_threshold_car", 0.55)
    detection_confidence_threshold_pedestrian = self.config.float("detection_confidence_threshold_pedestrian", 0.95)
    reid_weight_car = self.config.float("reid_weight_car", 1.0)
    reid_weight_pedestrian = self.config.float("reid_weight_pedestrian", 1.0)
    mask_iou_weight_car = self.config.float("mask_iou_weight_car", 0.0)
    mask_iou_weight_pedestrian = self.config.float("mask_iou_weight_pedestrian", 0.0)
    bbox_center_weight_car = self.config.float("bbox_center_weight_car", 0.0)
    bbox_center_weight_pedestrian = self.config.float("bbox_center_weight_pedestrian", 0.0)
    bbox_iou_weight_car = self.config.float("bbox_iou_weight_car", 0.0)
    bbox_iou_weight_pedestrian = self.config.float("bbox_iou_weight_pedestrian", 0.0)
    association_threshold_car = self.config.float("association_threshold_car", 0.3)
    association_threshold_pedestrian = self.config.float("association_threshold_pedestrian", 0.3)
    keep_alive_car = self.config.int("keep_alive_car", 0)
    keep_alive_pedestrian = self.config.int("keep_alive_pedestrian", 0)
    reid_euclidean_offset_car = self.config.float("reid_euclidean_offset_car", 5.0)
    reid_euclidean_scale_car = self.config.float("reid_euclidean_scale_car", 1.0)
    reid_euclidean_offset_pedestrian = self.config.float("reid_euclidean_offset_pedestrian", 5.0)
    reid_euclidean_scale_pedestrian = self.config.float("reid_euclidean_scale_pedestrian", 1.0)
    box_offset = self.config.float("box_offset", 50.0)
    box_scale = self.config.float("box_scale", 0.02)
    new_reid = self.config.bool("new_reid", False)
    new_reid_threshold_car = self.config.float("new_reid_threshold_car", 2.0)
    new_reid_threshold_pedestrian = self.config.float("new_reid_threshold_pedestrian", 2.0)
    self.tracker_options = {"tracker": tracker, "reid_comp": tracker_reid_comp,
                            "detection_confidence_threshold_car": detection_confidence_threshold_car,
                            "detection_confidence_threshold_pedestrian": detection_confidence_threshold_pedestrian,
                            "reid_weight_car": reid_weight_car,
                            "reid_weight_pedestrian": reid_weight_pedestrian,
                            "mask_iou_weight_car": mask_iou_weight_car,
                            "mask_iou_weight_pedestrian": mask_iou_weight_pedestrian,
                            "bbox_center_weight_car": bbox_center_weight_car,
                            "bbox_center_weight_pedestrian": bbox_center_weight_pedestrian,
                            "bbox_iou_weight_car": bbox_iou_weight_car,
                            "bbox_iou_weight_pedestrian": bbox_iou_weight_pedestrian,
                            "association_threshold_car": association_threshold_car,
                            "association_threshold_pedestrian": association_threshold_pedestrian,
                            "keep_alive_car": keep_alive_car,
                            "keep_alive_pedestrian": keep_alive_pedestrian,
                            "reid_euclidean_offset_car": reid_euclidean_offset_car,
                            "reid_euclidean_scale_car": reid_euclidean_scale_car,
                            "reid_euclidean_offset_pedestrian": reid_euclidean_offset_pedestrian,
                            "reid_euclidean_scale_pedestrian": reid_euclidean_scale_pedestrian,
                            "new_reid_threshold_car": new_reid_threshold_car,
                            "new_reid_threshold_pedestrian": new_reid_threshold_pedestrian,
                            "box_offset": box_offset,
                            "box_scale": box_scale,
                            "new_reid": new_reid}

    self.mask_disjoint_strategy = self.config.string("mask_disjoint_strategy", "y_pos")  # Or "score"

    self.export_detections = self.config.bool("export_detections", False)
    self.import_detections = self.config.bool("import_detections", False)
    self.visualize_tracks = self.config.bool("visualize_tracks", False)
    self.do_tracking = self.config.bool("do_tracking", True)
    self.embeddings = {}
    self.optical_flow_path = self.config.string("optical_flow_path",
                                                "/work/"+username()+"/data/KITTI_flow_pwc/")
    self.run_tracking_eval = self.config.bool("run_tracking_eval", False)

  def forward(self):
    super(TrackingForwarder, self).forward()
    if self.export_embeddings:
      # Export to tensorboard checkpoint
      out_folder = "forwarded/" + self.config.string("model") + "/embeddings"
      os.makedirs(out_folder, exist_ok=True)
      save_embeddings(self.embeddings, out_folder)
    if self.run_tracking_eval:
      #p = subprocess.run(["python3", "eval.py", "/home/pv182253/vision/savitar2/forwarded/conv3d/tracking_data/",
      #                    "/work/pv182253/data/MOTS/KITTI_MOTS/train/instances/", "val.seqmap"], stdout=subprocess.PIPE,
      #                   cwd="/home/pv182253/vision/mots_eval/")
      p = subprocess.run(["python3", "eval.py", "/home/voigtlaender/vision/savitar2/forwarded/" + self.config.string("model") + "/tracking_data/",
                          "/globalwork/voigtlaender/data/KITTI_MOTS/train/instances/", "val.seqmap"],
                         stdout=subprocess.PIPE, cwd="/home/voigtlaender/vision/mots_eval/")
      print(p.stdout.decode("utf-8"), file=log.v1)

  def _forward_video(self, n_timesteps, tag):
    print("Forwarding video...", file=log.v5)
    print(tag, file=log.v5)
    if self.import_detections:
      assert not self.export_embeddings
      assert not self.export_detections
      image_crops = None
      imgs = []
      print("Loading forwarded detections from file...", file=log.v5)
      time_start = perf_counter()
      det_boxes, det_scores, reid_features, det_classes, det_masks = \
        import_detections_for_sequence(tag, n_timesteps, self.config.string("detections_import_path", ""),
                                       self.config.string("model"), self.engine.start_epoch, self.add_masks)
      print("Done.", file=log.v5)
      if self.visualize_detections or self.visualize_tracks:
        print("Loading images for visualization...", file=log.v5)
        batch_size = self.val_data.get_batch_size()
        for t_start in range(0, n_timesteps, batch_size):
          dict = self.val_data.get_feed_dict_for_next_step()
          for j in range(batch_size):
            imgs.append(dict[self.val_data._placeholders[DataKeys.IMAGES][j]])
        print("Done.", file=log.v5)
    else:
      recurrent_state = None
      det_boxes = []
      det_scores = []
      det_classes = []
      det_masks = []
      reid_features = []
      imgs = []
      image_crops = []
      batch_size = self.val_data.get_batch_size()
      time_start = perf_counter()
      for t_start in range(0, n_timesteps, batch_size):
        print(t_start+1, "/", n_timesteps, file=log.v5)
        recurrent_state, measures, extractions = self._forward_timestep(recurrent_state)

        for j in range(batch_size):
          t = t_start + j
          if t >= n_timesteps:
            continue
          assert len(extractions[Extractions.DET_BOXES][0]) == batch_size, len(extractions[Extractions.DET_BOXES][0])
          det_boxes_t = extractions[Extractions.DET_BOXES][0][j]
          det_scores_t = extractions[Extractions.DET_PROBS][0][j]
          reid_features_t = extractions[Extractions.REID_FEATURES][0][j]
          det_classes_t = extractions[Extractions.DET_LABELS][0][j]
          if self.add_masks:
            if len(det_boxes_t) == 0:
              det_masks_t = []
            else:
              det_masks_t = [cocomask.encode(np.asfortranarray(m.squeeze(axis=0), dtype=np.uint8))
                             for m in np.vsplit(extractions[Extractions.DET_MASKS][0][j], len(det_boxes_t))]
          else:
            det_masks_t = [None] * len(det_boxes_t)
          det_boxes.append(det_boxes_t)
          det_scores.append(det_scores_t)
          reid_features.append(reid_features_t)
          det_classes.append(det_classes_t)
          det_masks.append(det_masks_t)
          if self.visualize_detections or self.visualize_tracks or self.export_embeddings:
            if DataKeys.RAW_IMAGES not in extractions:
              print("Can't extract raw images, maybe images in batch have different size?", file=log.v5)
              assert False
            img_t = extractions[DataKeys.RAW_IMAGES][0][j]
            imgs.append(img_t)
            if self.export_embeddings:
              det_boxes_t_i = det_boxes_t.astype(dtype=np.int32)
              for box in det_boxes_t_i:
                img_crop = imresize(img_t[box[1]:box[3], box[0]:box[2], :], size=(50, 50))
                image_crops.append(img_crop / 255.0)

    time_stop_fwd = perf_counter()
    if self.do_tracking:
      if self.tracker_options["mask_iou_weight_car"] > 0.0 or \
        self.tracker_options["mask_iou_weight_pedestrian"] > 0.0 or \
        self.tracker_options["bbox_iou_weight_car"] > 0.0 or \
        self.tracker_options["bbox_iou_weight_pedestrian"] > 0.0:
        optical_flow = load_optical_flow(tag, self.optical_flow_path)
      else:
        optical_flow = None
      hyp_tracks = track_single_sequence(self.tracker_options, det_boxes, det_scores, reid_features, det_classes,
                                         det_masks, optical_flow=optical_flow)
      if self.add_masks:
        hyp_tracks = self.make_disjoint_helper(hyp_tracks)
      time_stop_track = perf_counter()
      print("Time for tracking (s):", time_stop_track - time_stop_fwd, "FPS for tracking including forwarding:",
            n_timesteps / (time_stop_track - time_start), file=log.v5)
      print("Exporting tracking results", file=log.v5)
      time_starts_at_1 = False
      if hasattr(self.val_data, "time_starts_at_1") and self.val_data.time_starts_at_1:
          time_starts_at_1 = True
          print("Starting time at 1 for exporting", file=log.v1)
      export_tracking_result_in_kitti_format(tag, hyp_tracks, self.add_masks, self.config.string("model"),
                                             start_time_at_1=time_starts_at_1)
      if self.visualize_tracks:
        print("Visualizing tracks", file=log.v5)
        visualize_tracks(tag, hyp_tracks, imgs, self.add_masks,
                         self.interactive_visualization, self.config.string("model"))
    print("Time for forwarding (s):", time_stop_fwd - time_start, "FPS for forwarding (wo. tracking):",
          n_timesteps / (time_stop_fwd - time_start), file=log.v5)
    if self.export_detections:
      print("Exporting detections", file=log.v5)
      export_detections_for_sequence(tag, det_boxes, det_scores, reid_features, det_classes, det_masks,
                                     self.config.string("model"), self.engine.start_epoch, self.add_masks)
    if self.export_embeddings:
      print("Exporting embeddings", file=log.v5)
      # Save to export to tensorboard checkpoint
      image_crops = np.stack(image_crops, axis=0)
      embeddings = np.concatenate(reid_features, axis=0)
      labels = np.concatenate(det_classes, axis=0)
      self.embeddings[tag] = [image_crops, embeddings, labels]
    if self.visualize_detections:
      print("Visualizing detections", file=log.v5)
      visualize_detections_for_sequence(tag, det_boxes, det_scores, det_classes, det_masks, imgs,
                                        self.add_masks, self.interactive_visualization, self.config.string("model"))

  def make_disjoint_helper(self, tracks):
    return make_disjoint(tracks, self.mask_disjoint_strategy)


def visualize_detections_for_sequence(tag, det_boxes, det_scores, det_classes, det_masks, imgs,
                                      add_masks, interactive_visualization, model_str):
  if len(imgs) > len(det_boxes):
    print("warning, len of imgs and det_boxes does not match", len(imgs), len(det_boxes), file=log.v1)
    imgs = imgs[:len(det_boxes)]
  assert len(det_boxes) == len(imgs)
  for t, (boxes, scores, classes, masks, img) in enumerate(zip(det_boxes, det_scores, det_classes, det_masks, imgs)):
    if add_masks:
      masks_decoded = [cocomask.decode(m) for m in masks]
    else:
      masks_decoded = [None for _ in boxes]
    if interactive_visualization:
      out_filename = None
    else:
      out_folder = "forwarded/" + model_str + "/vis/detections/" + tag
      os.makedirs(out_folder, exist_ok=True)
      out_filename = out_folder + "/%06d.jpg" % t
    visualize_detections(boxes, classes, masks_decoded, scores, img, None, out_filename)


def visualize_tracks(tag, tracks, imgs, add_masks, interactive_visualization, model_str, box_is_xywh=False):
  if len(imgs) > len(tracks):
    print("warning, len of imgs and tracks does not match", len(imgs), len(tracks), file=log.v1)
    imgs = imgs[:len(tracks)]
  assert len(tracks) == len(imgs), (len(tracks), len(imgs))
  for t, (track, img) in enumerate(zip(tracks, imgs)):
    boxes = [te.box for te in track]
    classes = [te.class_ for te in track]
    if add_masks:
      masks = [cocomask.decode(te.mask) for te in track]
    else:
      masks = [None for _ in track]
    scores = [1.0 for _ in track]
    ids = [te.track_id for te in track]
    if interactive_visualization:
      out_filename = None
    else:
      out_folder = "forwarded/" + model_str + "/vis/tracks/" + tag
      os.makedirs(out_folder, exist_ok=True)
      out_filename = out_folder + "/%06d.jpg" % t
    visualize_detections(boxes, classes, masks, scores, img, ids, out_filename, box_is_xywh=box_is_xywh)
