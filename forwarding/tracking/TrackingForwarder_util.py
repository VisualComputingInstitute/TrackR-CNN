import glob
import os

import numpy as np
from pycocotools import mask as cocomask

from forwarding.tracking.Util_tracking import TrackElement


def make_disjoint(tracks, strategy):
  def get_max_y(obj):
    _, y, _, h = cocomask.toBbox(obj.mask)
    return y + h

  for frame, objects in enumerate(tracks):
    if len(objects) == 0:
      continue
    if strategy == "y_pos":
      objects_sorted = sorted(objects, key=lambda x: get_max_y(x), reverse=True)
    elif strategy == "score":
      objects_sorted = sorted(objects, key=lambda x: x.score, reverse=True)
    else:
      assert False, "Unknown mask_disjoint_strategy"
    objects_disjoint = [objects_sorted[0]]
    used_pixels = objects_sorted[0].mask
    for obj in objects_sorted[1:]:
      new_mask = obj.mask
      if cocomask.area(cocomask.merge([used_pixels, obj.mask], intersect=True)) > 0.0:
        obj_mask_decoded = cocomask.decode(obj.mask)
        used_pixels_decoded = cocomask.decode(used_pixels)
        obj_mask_decoded[np.where(used_pixels_decoded > 0)] = 0
        new_mask = cocomask.encode(obj_mask_decoded)
      used_pixels = cocomask.merge([used_pixels, obj.mask], intersect=False)
      objects_disjoint.append(TrackElement(box=obj.box, track_id=obj.track_id, class_=obj.class_, score=obj.score,
                                           mask=new_mask))
    tracks[frame] = objects_disjoint
  return tracks


def export_tracking_result_in_kitti_format(tag, tracks, add_masks, model_str, out_folder="", start_time_at_1=False):
  if out_folder == "":
    out_folder = "forwarded/" + model_str + "/tracking_data"
  os.makedirs(out_folder, exist_ok=True)
  out_filename = out_folder + "/" + tag + ".txt"
  with open(out_filename, "w") as f:
    start = 1 if start_time_at_1 else 0
    for t, tracks_t in enumerate(tracks, start):  # TODO this works?
      for track in tracks_t:
        if add_masks:
          print(t, track.track_id, track.class_,
                *track.mask['size'], track.mask['counts'].decode(encoding='UTF-8'), file=f)
        else:
          # TODO: class id to class name mapping atm harcoded
          if track.class_ == 1:
            class_str = "Car"
          elif track.class_ == 2:
            class_str = "Pedestrian"
          else:
            assert False, ("unknown class id", track.class_)
          print(t, track.track_id, class_str, -1, -1, -1, *track.box, -1, -1, -1, -1, -1, -1, -1, track.score, file=f)


def export_detections_for_sequence(tag, boxes, scores, reids, classes, masks, model_str, epoch, add_masks, out_folder=""):
  if out_folder == "":
    out_folder = "forwarded/" + model_str + "/detections/" + str(epoch)
  os.makedirs(out_folder, exist_ok=True)
  out_filename = out_folder + "/" + tag + ".txt"
  with open(out_filename, "w") as f:
    t = 0
    for boxes_t, scores_t, reids_t, classes_t, masks_t in zip(boxes, scores, reids, classes, masks):
      for box, score, reid, class_, mask in zip(boxes_t, scores_t, reids_t, classes_t, masks_t):
        if add_masks:
          print(t, *box, score, class_, *mask['size'], mask['counts'].decode(encoding='UTF-8'), *reid, file=f)
        else:
          print(t, *box, score, class_, *reid, file=f)
      t = t + 1


def import_detections_for_sequence(tag, max_len_seq, detections_import_path, model_str, epoch, add_masks,
                                   classes_to_load=None):
  if detections_import_path == "":
    detections_import_path = "forwarded/" + model_str + "/detections/" + str(epoch)
  out_filename = detections_import_path + "/" + tag + ".txt"
  with open(out_filename) as f:
    content = f.readlines()
  boxes = []
  scores = []
  reids = []
  classes = []
  masks = []
  for line in content:
    entries = line.split(' ')
    # filter to classes_to_load if it is specified
    if classes_to_load is not None and int(entries[6]) not in classes_to_load:
      continue
    t = int(entries[0])
    while t+1 > len(boxes):
      boxes.append([])
      scores.append([])
      reids.append([])
      classes.append([])
      masks.append([])
    boxes[t].append([float(entries[1]), float(entries[2]), float(entries[3]), float(entries[4])])
    scores[t].append(float(entries[5]))
    classes[t].append(int(entries[6]))
    if add_masks:
      masks[t].append({'size': [int(entries[7]), int(entries[8])],
                       'counts': entries[9].strip().encode(encoding='UTF-8')})
      reids[t].append([float(e) for e in entries[10:]])
    else:
      masks[t].append([])
      reids[t].append([float(e) for e in entries[7:]])

  while max_len_seq > len(boxes):
    boxes.append([])
    scores.append([])
    reids.append([])
    classes.append([])
    masks.append([])

  # transform into numpy arrays
  for t in range(len(boxes)):
    if len(boxes[t]) > 0:
      boxes[t] = np.vstack(boxes[t])
      scores[t] = np.array(scores[t])
      classes[t] = np.array(classes[t])
      reids[t] = np.vstack(reids[t])
  return boxes, scores, reids, classes, masks


def load_optical_flow(tag, optical_flow_path):
  import pickle
  if os.path.exists(optical_flow_path + "/preprocessed_" + tag):
    with open(optical_flow_path + "/preprocessed_" + tag, 'rb') as input:
      flows = pickle.load(input)
  else:
    flow_files_x = sorted(glob.glob(optical_flow_path + "/" + tag + "/*_x_minimal*.png"))
    flow_files_y = sorted(glob.glob(optical_flow_path + "/" + tag + "/*_y_minimal*.png"))
    assert len(flow_files_x) == len(flow_files_y)
    flows = [open_flow_png_file([x, y]) for x, y in zip(flow_files_x, flow_files_y)]
    with open(optical_flow_path + "/preprocessed_" + tag, 'wb') as output:
      pickle.dump(flows, output, pickle.HIGHEST_PROTOCOL)
  return flows


def open_flow_png_file(file_path_list):
  # Funtion from Kilian Merkelbach.
  # Decode the information stored in the filename
  flow_png_info = {}
  assert len(file_path_list) == 2
  for file_path in file_path_list:
    file_token_list = os.path.splitext(file_path)[0].split("_")
    minimal_value = int(file_token_list[-1].replace("minimal", ""))
    flow_axis = file_token_list[-2]
    flow_png_info[flow_axis] = {'path': file_path,
                                'minimal_value': minimal_value}

  # Open both files and add back the minimal value
  for axis, flow_info in flow_png_info.items():
    import png
    png_reader = png.Reader(filename=flow_info['path'])
    flow_2d = np.vstack(map(np.uint16, png_reader.asDirect()[2]))

    # Add the minimal value back
    flow_2d = flow_2d.astype(np.int16) + flow_info['minimal_value']

    flow_png_info[axis]['flow'] = flow_2d

  # Combine the flows
  flow_x = flow_png_info['x']['flow']
  flow_y = flow_png_info['y']['flow']
  flow = np.stack([flow_x, flow_y], 2)

  return flow