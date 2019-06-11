import tensorflow as tf

from core.Log import log
from core import Measures, Extractions
from datasets import DataKeys
from forwarding.Forwarder import Forwarder
from forwarding.Util import apply_mask
import gc
import colorsys
import numpy as np


DETECTION_EXTRACTION_KEYS = (Extractions.DET_BOXES, Extractions.DET_PROBS, Extractions.DET_LABELS,
                             DataKeys.IMAGE_FILENAMES, Extractions.DET_MASKS, DataKeys.RAW_IMAGES,)

# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors():
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  N = 30
  brightness = 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6, 10]
  colors = [colors[idx] for idx in perm]
  return colors

class RecurrentDetectionForwarder(Forwarder):
  def __init__(self, engine, extraction_keys=DETECTION_EXTRACTION_KEYS):
    super().__init__(engine)
    self.video_ids = self.config.int_list("video_ids", [])  # Which vids to forward
    self.visualize_detections = self.config.bool("visualize_detections", False)
    # Only relevant if visualize_detections==True, if interactive show  during forwarding instead of exporting jpgs
    self.interactive_visualization = self.config.bool("interactive_visualization", False)
    # Output for KITTI eval
    self.export_for_kitti_semseg = self.config.bool("export_for_kitti_semseg", True)
    self._extraction_keys = extraction_keys
    if Extractions.RECURRENT_STATE not in self._extraction_keys:
      self._extraction_keys += (Extractions.RECURRENT_STATE,)

  def forward(self):
    video_ids = range(self.val_data.n_videos()) if len(self.video_ids) == 0 else self.video_ids
    for video_idx in video_ids:
      self.val_data.set_video_idx(video_idx)
      gc.collect()
      tag = self.val_data.get_video_tag()
      n_timesteps = self.val_data.n_examples_per_epoch()
      self._forward_video(n_timesteps, tag)
      gc.collect()

  def _forward_video(self, n_timesteps, tag):
    recurrent_states = None
    batch_size = self.val_data.get_batch_size()
    for t_start in range(0, n_timesteps, batch_size):
      recurrent_state, measures, extractions = self._forward_timestep(recurrent_states)

      for j in range(batch_size):
        t = t_start + j
        if t >= n_timesteps:
          continue
        det_boxes = extractions[Extractions.DET_BOXES][0][j]
        det_scores = extractions[Extractions.DET_PROBS][0][j]
        det_classes = extractions[Extractions.DET_LABELS][0][j]
        det_masks = extractions[Extractions.DET_MASKS][0][j]
        img_filename = extractions[DataKeys.IMAGE_FILENAMES][0][j].decode("utf-8")
        print(img_filename)
        if self.export_for_kitti_semseg:
          out_folder = "forwarded/" + self.config.string("model") + "/seg_data/"
          _export_detections_kitti_format(out_folder, det_boxes, det_classes, det_masks, det_scores, t, tag)
        if self.visualize_detections:
          if DataKeys.RAW_IMAGES not in extractions:
            print("Can't extract raw images for visualization, maybe images in batch have different size?", file=log.v5)
            assert False
          img = extractions[DataKeys.RAW_IMAGES][0][j]
          if self.interactive_visualization:
            out_filename = None
          else:
            out_folder = "forwarded/" + self.config.string("model") + "/vis/" + tag
            tf.gfile.MakeDirs(out_folder)
            out_filename = out_folder + "/%06d.jpg" % t
          visualize_detections(det_boxes, det_classes, det_masks, det_scores, img, save_path=out_filename)

  def _forward_timestep(self, recurrent_states):
    feed_dict = self.val_data.get_feed_dict_for_next_step()
    if recurrent_states is not None:
      placeholders = self.engine.test_network.placeholders
      assert len(placeholders) == len(recurrent_states)
      for placeholder, val in zip(placeholders, recurrent_states):
        feed_dict[placeholder] = val
    step_res = self.trainer.validation_step(feed_dict=feed_dict,
                                            extraction_keys=self._extraction_keys)
    measures = step_res[Measures.MEASURES]
    extractions = step_res[Extractions.EXTRACTIONS]
    if Extractions.RECURRENT_STATE in extractions:
      recurrent_states = extractions[Extractions.RECURRENT_STATE]
      assert len(recurrent_states) == 1
      recurrent_states = recurrent_states[0]
      # flatten
      recurrent_states = [item for statetuple in recurrent_states for item in statetuple]
      recurrent_states = [x[np.newaxis] if len(x.shape) == 1 else x for x in recurrent_states]
    return recurrent_states, measures, extractions


def _export_detections_kitti_format(out_folder, det_boxes, det_classes, det_masks, det_scores, t, tag):
  assert len(det_boxes) == len(det_scores) == len(det_classes) == len(det_masks)

  # Format for evalInstanceLevelSemanticLabeling.py from KITTI eval suite (with our changes)

  # Create folders
  tf.gfile.MakeDirs(out_folder + "pred_list/")
  tf.gfile.MakeDirs(out_folder + "pred_img/")

  image_name = tag + "_" + str(t).zfill(6)
  # Create file with list of detections + id
  with open(out_folder + "pred_list/" + image_name + ".txt", "w+") as f:
    for idx, (bbox, score, class_, mask) in enumerate(zip(det_boxes, det_scores, det_classes, det_masks)):
      mask_img_filename = "pred_img/" + image_name + "_" + str(idx).zfill(3) + ".png"
      from PIL import Image
      im = Image.fromarray(mask * 255)
      im.save(out_folder + mask_img_filename)
      f.write("../" + mask_img_filename + " " + str(class_) + " " + str(score) + "\n")


def visualize_detections(det_boxes, det_classes, det_masks, det_scores, img, ids=None, save_path=None,
                         draw_boxes=False, box_is_xywh=False):
  colors = generate_colors()
  if save_path is not None:
    import matplotlib
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig = plt.figure()
  dpi = 100.0
  fig.set_size_inches(img.shape[1]/dpi, img.shape[0]/dpi, forward=True)
  fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
  ax = fig.subplots()
  ax.set_axis_off()
  assert len(det_boxes) == len(det_scores) == len(det_classes) == len(det_masks)
  for idx, (bbox, score, class_, mask) in enumerate(zip(det_boxes, det_scores, det_classes, det_masks)):
    if ids is None:
      color = colors[idx % len(colors)]
    else:
      color = colors[ids[idx] % len(colors)]
    # TODO
    if class_ == 1:
      category_name = "Car"
    elif class_ == 2:
      category_name = "Pedestrian"
    else:
      category_name = "Ignore"
      color = (0.7, 0.7, 0.7)

    if class_ == 1 or class_ == 2:  # Don't show boxes or ids for ignore regions
      if ids is not None:
        category_name += ":" + str(ids[idx])
      if score < 1.0:
        category_name += ":" + "%.2f" % score
      if not box_is_xywh:
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
      if draw_boxes:
        import matplotlib.patches as patches
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                                 edgecolor=color, facecolor='none', alpha=1.0)
        ax.add_patch(rect)
      ax.annotate(category_name, (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]), color=color, weight='bold',
                  fontsize=7, ha='center', va='center', alpha=1.0)
    if mask is not None:
      apply_mask(img, mask, color, alpha=score * 0.5)
  ax.imshow(img)
  if save_path is None:
    plt.show()
  else:
    print(save_path, file=log.v5)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
