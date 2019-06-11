# a lot of stuff is copied over from savitar1, but we still need to port more of it

import numpy as np
from scipy.special import expit as sigmoid
from collections import namedtuple
import munkres
from scipy.spatial.distance import cdist
import pycocotools.mask as cocomask
from cv2 import remap, INTER_NEAREST

TrackElement_ = namedtuple("TrackElement", ["t", "box", "reid", "track_id", "class_", "mask", "score"])
TrackElement = namedtuple("TrackElement", ["box", "track_id", "class_", "mask", "score"])

munkres_obj = munkres.Munkres()


def track_single_sequence(tracker_options, boxes, scores, reids, classes, masks, optical_flow=None):
  # perform tracking per class and in the end combine the results
  classes_flat = [c for cs in classes for c in cs]
  unique_classes = np.unique(classes_flat)
  start_track_id = 1
  class_tracks = []
  tracker_options_class = {"tracker": tracker_options["tracker"], "reid_comp": tracker_options["reid_comp"],
                           "box_offset": tracker_options["box_offset"],
                           "box_scale": tracker_options["box_scale"]}
  for class_ in unique_classes:
    if class_ == 1:
      tracker_options_class["detection_confidence_threshold"] = tracker_options["detection_confidence_threshold_car"]
      tracker_options_class["reid_weight"] = tracker_options["reid_weight_car"]
      tracker_options_class["mask_iou_weight"] = tracker_options["mask_iou_weight_car"]
      tracker_options_class["bbox_iou_weight"] = tracker_options["bbox_iou_weight_car"]
      tracker_options_class["bbox_center_weight"] = tracker_options["bbox_center_weight_car"]
      tracker_options_class["association_threshold"] = tracker_options["association_threshold_car"]
      tracker_options_class["keep_alive"] = tracker_options["keep_alive_car"]
      tracker_options_class["new_reid_threshold"] = tracker_options["new_reid_threshold_car"]
      tracker_options_class["reid_euclidean_offset"] = tracker_options["reid_euclidean_offset_car"]
      tracker_options_class["reid_euclidean_scale"] = tracker_options["reid_euclidean_scale_car"]
    elif class_ == 2:
      tracker_options_class["detection_confidence_threshold"] = tracker_options[
        "detection_confidence_threshold_pedestrian"]
      tracker_options_class["reid_weight"] = tracker_options["reid_weight_pedestrian"]
      tracker_options_class["mask_iou_weight"] = tracker_options["mask_iou_weight_pedestrian"]
      tracker_options_class["bbox_iou_weight"] = tracker_options["bbox_iou_weight_pedestrian"]
      tracker_options_class["bbox_center_weight"] = tracker_options["bbox_center_weight_pedestrian"]
      tracker_options_class["association_threshold"] = tracker_options["association_threshold_pedestrian"]
      tracker_options_class["keep_alive"] = tracker_options["keep_alive_pedestrian"]
      tracker_options_class["new_reid_threshold"] = tracker_options["new_reid_threshold_pedestrian"]
      tracker_options_class["reid_euclidean_offset"] = tracker_options["reid_euclidean_offset_pedestrian"]
      tracker_options_class["reid_euclidean_scale"] = tracker_options["reid_euclidean_scale_pedestrian"]
    else:
      assert False, "unknown class"
    if tracker_options["new_reid"]:
      tracks = tracker_per_class_new_reid(tracker_options_class, boxes, scores, reids, classes, masks, class_,
                                          start_track_id, optical_flow=optical_flow)
    else:
      tracks = tracker_per_class(tracker_options_class, boxes, scores, reids, classes, masks, class_, start_track_id,
                                 optical_flow=optical_flow)
    class_tracks.append(tracks)
    track_ids_flat = [track.track_id for tracks_t in tracks for track in tracks_t]
    track_ids_flat.append(start_track_id)
    start_track_id = max(track_ids_flat) + 1

  n_timesteps = len(boxes)
  tracks_combined = [[] for _ in range(n_timesteps)]
  for tracks_c in class_tracks:
    for t, tracks_c_t in enumerate(tracks_c):
      tracks_combined[t].extend(tracks_c_t)
  return tracks_combined


def tracker_per_class(tracker_options, boxes, scores, reids, classes, masks, class_to_track, start_track_id,
                      optical_flow=None):
  max_track_id = start_track_id
  all_tracks = []
  active_tracks = []
  if optical_flow is None:
    optical_flow = [None for _ in masks]
  else:
    optical_flow = [None] + optical_flow
    assert len(boxes) == len(scores) == len(reids) == len(classes) == len(masks) == len(optical_flow)
  for t, (boxes_t, scores_t, reids_t, classes_t, masks_t, flow_tm1_t) in enumerate(zip(boxes, scores, reids,
                                                                                   classes, masks, optical_flow)):
    detections_t = []
    for box, score, reid, class_, mask in zip(boxes_t, scores_t, reids_t, classes_t, masks_t):
      if class_ != class_to_track:
        continue
      if mask is not None and cocomask.area(mask) <= 10:
        continue
      if score >= tracker_options["detection_confidence_threshold"]:
        detections_t.append((box, reid, mask, class_, score))
      else:
        continue
    if len(detections_t) == 0:
      curr_tracks = []
    elif len(active_tracks) == 0:
      curr_tracks = []
      for det in detections_t:
        curr_tracks.append(TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                                         track_id=max_track_id, score=det[4]))
        max_track_id += 1
    else:
      association_similarities = np.zeros((len(detections_t), len(active_tracks)))

      if tracker_options["reid_weight"] != 0:
        curr_reids = np.array([x[1] for x in detections_t], dtype="float64")
        last_reids = np.array([x.reid for x in active_tracks], dtype="float64")
        if tracker_options["reid_comp"] == "sigmoid_dot":
          reid_similarities = sigmoid(np.dot(curr_reids, last_reids.T))
        elif tracker_options["reid_comp"] == "cosine":
          reid_similarities = np.dot(curr_reids/np.linalg.norm(curr_reids, axis=1, ord=2)[:, np.newaxis],
                                     (last_reids/np.linalg.norm(last_reids, axis=1, ord=2)[:, np.newaxis]).T)
        elif tracker_options["reid_comp"] == "euclidean":
          reid_dists = cdist(curr_reids, last_reids, "euclidean")
          reid_similarities = tracker_options["reid_euclidean_scale"] *\
                              (tracker_options["reid_euclidean_offset"] - reid_dists)
        elif tracker_options["reid_comp"] == "normalized_euclidean":
          reid_dists = cdist(curr_reids/np.linalg.norm(curr_reids, axis=1, ord=2)[:, np.newaxis],
                             last_reids/np.linalg.norm(last_reids, axis=1, ord=2)[:, np.newaxis], "euclidean")
          reid_similarities = 1 - reid_dists
        else:
          assert False
        association_similarities += tracker_options["reid_weight"] * reid_similarities

      if tracker_options["mask_iou_weight"] != 0:
        # Prepare flow
        h, w = flow_tm1_t.shape[:2]
        flow_tm1_t = -flow_tm1_t
        flow_tm1_t[:, :, 0] += np.arange(w)
        flow_tm1_t[:, :, 1] += np.arange(h)[:, np.newaxis]
        masks_t = [v[2] for v in detections_t]
        masks_tm1 = [v.mask for v in active_tracks]
        masks_tm1_warped = [warp_flow(mask, flow_tm1_t) for mask in masks_tm1]
        mask_ious = cocomask.iou(masks_t, masks_tm1_warped, [False] * len(masks_tm1_warped))
        association_similarities += tracker_options["mask_iou_weight"] * mask_ious

      if tracker_options["bbox_center_weight"] != 0:
        centers_t = [v[0][0:2] + (v[0][2:4] - v[0][0:2]) / 2 for v in detections_t]
        centers_tm1 = [v.box[0:2] + (v.box[2:4] - v.box[0:2]) / 2 for v in active_tracks]
        box_dists = cdist(np.array(centers_t), np.array(centers_tm1), "euclidean")
        box_similarities = tracker_options["box_scale"] *\
                           (tracker_options["box_offset"] - box_dists)
        association_similarities += tracker_options["bbox_center_weight"] * box_similarities

      if tracker_options["bbox_iou_weight"] != 0:
        bboxes_t = [v[0] for v in detections_t]
        bboxes_tm1 = [v.box for v in active_tracks]
        bboxes_tm1_warped = [warp_box(box, flow_tm1_t) for box in bboxes_tm1]
        bbox_ious = np.array([[bbox_iou(box1, box2) for box1 in bboxes_tm1_warped] for box2 in bboxes_t])
        assert (0 <= bbox_ious).all() and (bbox_ious <= 1).all()
        association_similarities += tracker_options["bbox_iou_weight"] * bbox_ious

      curr_tracks = []
      detections_assigned = [False for _ in detections_t]
      if tracker_options["tracker"] == "greedy":
        while True:
          idx = association_similarities.argmax()
          idx = np.unravel_index(idx, association_similarities.shape)
          val = association_similarities[idx]
          if val < tracker_options["association_threshold"]:
            break
          det = detections_t[idx[0]]
          te = TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                             track_id=active_tracks[idx[1]].track_id, score=det[4])
          curr_tracks.append(te)
          detections_assigned[idx[0]] = True
          association_similarities[idx[0], :] = -1e10
          association_similarities[:, idx[1]] = -1e10
      elif tracker_options["tracker"] == "hungarian":
        cost_matrix = munkres.make_cost_matrix(association_similarities)
        disallow_indices = np.argwhere(association_similarities <= tracker_options["association_threshold"])
        for ind in disallow_indices:
          cost_matrix[ind[0]][ind[1]] = 1e9
        indexes = munkres_obj.compute(cost_matrix)
        for row, column in indexes:
          value = cost_matrix[row][column]
          if value == 1e9:
            continue
          det = detections_t[row]
          te = TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                             track_id=active_tracks[column].track_id, score=det[4])
          curr_tracks.append(te)
          detections_assigned[row] = True
      else:
        assert False
      for det, assigned in zip(detections_t, detections_assigned):
        if not assigned:
          curr_tracks.append(TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                                           track_id=max_track_id, score=det[4]))
          max_track_id += 1
    all_tracks.append(curr_tracks)
    newly_active_ids = {track.track_id for track in curr_tracks}
    active_tracks = [track for track in active_tracks
                     if track.track_id not in newly_active_ids and track.t >= t - tracker_options["keep_alive"]]
    active_tracks.extend(curr_tracks)
  # remove the reid values, since they are an implementation detail of the tracker and should not be part of the result
  result = [[TrackElement(box=track.box, track_id=track.track_id, mask=track.mask, class_=track.class_, score=track.score)
             for track in tracks_t] for tracks_t in all_tracks]
  return result


def tracker_per_class_new_reid(tracker_options, boxes, scores, reids, classes, masks, class_to_track, start_track_id,
                               optical_flow=None):
  assert tracker_options["reid_comp"] == "euclidean"
  assert tracker_options["tracker"] == "hungarian"
  max_track_id = start_track_id
  all_tracks = []
  last_tracks = []
  if optical_flow is None:
    optical_flow = [None for _ in masks]
  else:
    optical_flow = [None] + optical_flow
    assert len(boxes) == len(scores) == len(reids) == len(classes) == len(masks) == len(optical_flow)
  for t, (boxes_t, scores_t, reids_t, classes_t, masks_t, flow_tm1_t) in enumerate(zip(boxes, scores, reids,
                                                                                   classes, masks, optical_flow)):
    curr_tracks = []
    assigned_track_ids = []
    all_detections_t = []
    ### build all_detections_t
    for box, score, reid, class_, mask in zip(boxes_t, scores_t, reids_t, classes_t, masks_t):
      if class_ != class_to_track:
        continue
      if mask is not None and cocomask.area(mask) <= 10:
        continue
      all_detections_t.append((box, reid, mask, class_, score))

    # assign high confidence dets by association scores
    high_confidence_detections_t = [d for d in all_detections_t if
                                    d[4] >= tracker_options["detection_confidence_threshold"]]
    detections_assigned = [False for _ in high_confidence_detections_t]
    if len(high_confidence_detections_t) > 0 and len(last_tracks) > 0:
      association_similarities = calculate_association_similarities(high_confidence_detections_t, last_tracks,
                                                                    flow_tm1_t, tracker_options)
      cost_matrix = munkres.make_cost_matrix(association_similarities)
      disallow_indices = np.argwhere(association_similarities <= tracker_options["association_threshold"])
      for ind in disallow_indices:
        cost_matrix[ind[0]][ind[1]] = 1e9
      indexes = munkres_obj.compute(cost_matrix)
      for row, column in indexes:
        value = cost_matrix[row][column]
        if value == 1e9:
          continue
        det = high_confidence_detections_t[row]
        track_id = last_tracks[column].track_id
        te = TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                           track_id=track_id, score=det[4])
        assigned_track_ids.append(track_id)
        curr_tracks.append(te)
        detections_assigned[row] = True

    #### begin reid stuff ####
    old_tracks = []
    for tracks_in_time_step in all_tracks:
      for track_obj in tracks_in_time_step:
        if track_obj.track_id not in assigned_track_ids:
          old_tracks.append(track_obj)
    old_reids = np.array([x.reid for x in old_tracks], dtype="float32")
    # low conf dets
    dets_for_reid = [d for d in all_detections_t if d[4] < tracker_options["detection_confidence_threshold"]]
    # use unassigned high conf dets as well?
    for det, assigned in zip(high_confidence_detections_t, detections_assigned):
      if not assigned:
        dets_for_reid.append(det)
    curr_reids = np.array([d[1] for d in dets_for_reid], dtype="float32")
    reided_dets = []
    if old_reids.size > 0 and curr_reids.size > 0:
      reid_dists = cdist(curr_reids, old_reids, "euclidean")
      while True:
        idx = reid_dists.argmin()
        idx = np.unravel_index(idx, reid_dists.shape)
        val = reid_dists[idx]
        if val > tracker_options["new_reid_threshold"]:
          break
        #print("reided", class_to_track, val)
        det = dets_for_reid[idx[0]]
        reided_dets.append(det)
        track = old_tracks[idx[1]]
        te = TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                           track_id=track.track_id, score=det[4])
        curr_tracks.append(te)
        reid_dists[idx[0], :] = 1e10
        for idx, track2 in enumerate(old_tracks):
          if track.track_id == track2.track_id:
            reid_dists[:, idx] = 1e10
    ### end reid stuff ###

    # assign every high confidence det which has neither been propagated nor reided to a new track
    for det, assigned in zip(high_confidence_detections_t, detections_assigned):
      if not assigned:
        curr_tracks.append(TrackElement_(t=t, box=det[0], reid=det[1], mask=det[2], class_=det[3],
                                         track_id=max_track_id, score=det[4]))
        max_track_id += 1

    all_tracks.append(curr_tracks)
    last_tracks = curr_tracks


  # remove the reid values, since they are an implementation detail of the tracker and should not be part of the result
  result = [[TrackElement(box=track.box, track_id=track.track_id, mask=track.mask, class_=track.class_, score=track.score)
             for track in tracks_t] for tracks_t in all_tracks]
  return result


def calculate_association_similarities(detections_t, last_tracks, flow_tm1_t, tracker_options):
  association_similarities = np.zeros((len(detections_t), len(last_tracks)))
  if tracker_options["reid_weight"] != 0:
    curr_reids = np.array([x[1] for x in detections_t], dtype="float64")
    last_reids = np.array([x.reid for x in last_tracks], dtype="float64")
    reid_dists = cdist(curr_reids, last_reids, "euclidean")
    reid_similarities = tracker_options["reid_euclidean_scale"] * \
                        (tracker_options["reid_euclidean_offset"] - reid_dists)
    association_similarities += tracker_options["reid_weight"] * reid_similarities
  if tracker_options["mask_iou_weight"] != 0:
    # Prepare flow
    h, w = flow_tm1_t.shape[:2]
    flow_tm1_t = -flow_tm1_t
    flow_tm1_t[:, :, 0] += np.arange(w)
    flow_tm1_t[:, :, 1] += np.arange(h)[:, np.newaxis]
    masks_t = [v[2] for v in detections_t]
    masks_tm1 = [v.mask for v in last_tracks]
    masks_tm1_warped = [warp_flow(mask, flow_tm1_t) for mask in masks_tm1]
    mask_ious = cocomask.iou(masks_t, masks_tm1_warped, [False] * len(masks_tm1_warped))
    association_similarities += tracker_options["mask_iou_weight"] * mask_ious
  if tracker_options["bbox_center_weight"] != 0:
    centers_t = [v[0][0:2] + (v[0][2:4] - v[0][0:2]) / 2 for v in detections_t]
    centers_tm1 = [v.box[0:2] + (v.box[2:4] - v.box[0:2]) / 2 for v in last_tracks]
    box_dists = cdist(np.array(centers_t), np.array(centers_tm1), "euclidean")
    box_similarities = tracker_options["box_scale"] * \
                       (tracker_options["box_offset"] - box_dists)
    association_similarities += tracker_options["bbox_center_weight"] * box_similarities
  if tracker_options["bbox_iou_weight"] != 0:
    bboxes_t = [v[0] for v in detections_t]
    bboxes_tm1 = [v.box for v in last_tracks]
    bboxes_tm1_warped = [warp_box(box, flow_tm1_t) for box in bboxes_tm1]
    bbox_ious = np.array([[bbox_iou(box1, box2) for box1 in bboxes_tm1_warped] for box2 in bboxes_t])
    assert (0 <= bbox_ious).all() and (bbox_ious <= 1).all()
    association_similarities += tracker_options["bbox_iou_weight"] * bbox_ious
  return association_similarities


def warp_flow(mask_as_rle, flow):
  # unpack
  mask = cocomask.decode([mask_as_rle])
  # warp
  warped = _warp(mask, flow)
  # pack
  packed = cocomask.encode(np.asfortranarray(warped))
  return packed


def _warp(img, flow):
  # for some reason the result is all zeros with INTER_LINEAR...
  # res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
  res = remap(img, flow, None, INTER_NEAREST)
  res = np.equal(res, 1).astype(np.uint8)
  return res


def bbox_iou(box1, box2):
  x0_min = min(box1[0], box2[0])
  x0_max = max(box1[0], box2[0])
  y0_min = min(box1[1], box2[1])
  y0_max = max(box1[1], box2[1])
  x1_min = min(box1[2], box2[2])
  x1_max = max(box1[2], box2[2])
  y1_min = min(box1[3], box2[3])
  y1_max = max(box1[3], box2[3])
  I = max(x1_min - x0_max, 0) * max(y1_min - y0_max, 0)
  U = (x1_max - x0_min) * (y1_max - y0_min)
  if U == 0:
    return 0.0
  else:
    return I / U


def warp_box(box, flow):
  box_rounded = np.maximum(box.round().astype("int32"), 0)
  x0, y0, x1, y1 = box_rounded
  flows = flow[y0:y1, x0:x1]
  flows_x = flows[:, :, 0]
  flows_y = flows[:, :, 1]
  flow_x = np.median(flows_x)
  flow_y = np.median(flows_y)
  box_warped = box + [flow_x, flow_y, flow_x, flow_y]
  return box_warped
