import tensorflow as tf
import numpy as np
from PIL import Image

from core.Util import calculate_ious, calculate_ioas
from datasets import DataKeys
from datasets.util.BoundingBox import get_bbox_from_segmentation_mask_np
from functools import partial
from skimage.transform import integral_image


# TODO: clean this up
# These will be set init init_detection
ALL_ANCHORS = None
anchorH, anchorW = None, None

# From https://raw.githubusercontent.com/ppwwyyxx/tensorpack/master/examples/FasterRCNN/config.py
NUM_ANCHOR = 15  # This is hard coded
ANCHOR_STRIDE = 16
# TODO maybe read all of these from config too
POSITIVE_ANCHOR_THRES = 0.7
NEGATIVE_ANCHOR_THRES = 0.3
CROWD_OVERLAP_THRES = 0.7
RPN_FG_RATIO = 0.5
RPN_BATCH_PER_IM = 256


def init_anchors(config):
  task = config.string("task", "train")
  need_val = task != "train_no_val"
  max_size = max(config.int_list("input_size_train", []))
  if need_val and not config.int_list("input_size_val", []) == []:
    max_size = max(config.int_list("input_size_val", []))
  global ALL_ANCHORS
  ALL_ANCHORS = get_all_anchors(max_size)
  global anchorH, anchorW
  anchorH, anchorW = ALL_ANCHORS.shape[:2]


# From https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/utils/generate_anchors.py
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """

  base_anchor = np.array([1, 1, base_size, base_size], dtype='float32') - 1
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
  return anchors


def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  return anchors


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

  return anchors


# From https://raw.githubusercontent.com/ppwwyyxx/tensorpack/master/examples/FasterRCNN/data.py
def get_all_anchors(
    MAX_SIZE,
    stride=ANCHOR_STRIDE,
    sizes=(32, 64, 128, 256, 512),
    ratios=(0.5, 1., 2.)):
  """
  Get all anchors in the largest possible image, shifted, floatbox

  Returns:
      anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
      The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SCALE.

  """
  # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
  # are centered on stride / 2, have (approximate) sqrt areas of the specified
  # sizes, and aspect ratios as given.
  cell_anchors = generate_anchors(
    stride,
    scales=np.array(sizes, dtype=np.float) / stride,
    ratios=ratios)
  # anchors are intbox here.
  # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

  field_size = int(np.ceil(MAX_SIZE / stride))
  shifts = np.arange(0, field_size) * stride
  shift_x, shift_y = np.meshgrid(shifts, shifts)
  shift_x = shift_x.flatten()
  shift_y = shift_y.flatten()
  shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
  # Kx4, K = field_size * field_size
  K = shifts.shape[0]

  A = cell_anchors.shape[0]
  field_of_anchors = (
      cell_anchors.reshape((1, A, 4)) +
      shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
  field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
  # FSxFSxAx4
  #assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
  field_of_anchors = field_of_anchors.astype('float32')
  field_of_anchors[:, :, :, [2, 3]] += 1
  return field_of_anchors


def calculate_ioas_using_ignore_mask(bboxes1, ignore_mask):
  ignore_mask = (ignore_mask > 0).astype(np.float32)
  ignore_mask_cum_sum = integral_image(ignore_mask)
  ignore_mask_cum_sum = np.pad(ignore_mask_cum_sum, ((1, 0), (1, 0)), 'constant', constant_values=0)
  A = ignore_mask_cum_sum[-1, -1]
  x1, y1, x2, y2 = np.split(bboxes1.astype(np.int32), 4, axis=1)
  x1 = np.clip(x1, 0, ignore_mask.shape[1])
  y1 = np.clip(y1, 0, ignore_mask.shape[0])
  x2 = np.clip(x2, 0, ignore_mask.shape[1])
  y2 = np.clip(y2, 0, ignore_mask.shape[0])
  Is = ignore_mask_cum_sum[y2, x2] - ignore_mask_cum_sum[y2, x1] -\
       ignore_mask_cum_sum[y1, x2] + ignore_mask_cum_sum[y1, x1]
  return Is / A


def get_anchor_labels(anchors, gt_boxes, crowd_boxes, prefer_gt_to_ignore=False, use_ioa_for_ignore=False,
                      use_masks_for_ignore=False, crowd_masks=None):
  """
  Label each anchor as fg/bg/ignore.
  Args:
      anchors: Ax4 float
      gt_boxes: Bx4 float
      crowd_boxes: Cx4 float

  Returns:
      anchor_labels: (A,) int. Each element is {-1, 0, 1}
      anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
  """

  # This function will modify labels and return the filtered inds
  def filter_box_label(labels, value, max_num):
    curr_inds = np.where(labels == value)[0]
    if len(curr_inds) > max_num:
      disable_inds = np.random.choice(
        curr_inds, size=(len(curr_inds) - max_num),
        replace=False)
      labels[disable_inds] = -1  # ignore them
      curr_inds = np.where(labels == value)[0]
    return curr_inds

  NA, NB = len(anchors), len(gt_boxes)
  if NB <= 0:  # empty images should have been filtered already
    anchor_labels = np.ones((NA,), dtype='int32')
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    return anchor_labels, anchor_boxes, True

  box_ious = calculate_ious(anchors, gt_boxes)  # NA x NB
  ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
  ious_max_per_anchor = box_ious.max(axis=1)
  ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
  # for each gt, find all those anchors (including ties) that has the max ious with it
  anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

  # Setting NA labels: 1--fg 0--bg -1--ignore
  anchor_labels = -np.ones((NA,), dtype='int32')  # NA,

  # the order of setting neg/pos labels matter
  anchor_labels[anchors_with_max_iou_per_gt] = 1
  anchor_labels[ious_max_per_anchor >= POSITIVE_ANCHOR_THRES] = 1
  anchor_labels[ious_max_per_anchor < NEGATIVE_ANCHOR_THRES] = 0

  # First label all non-ignore candidate boxes which overlap crowd as ignore
  if crowd_boxes.size > 0:
    if prefer_gt_to_ignore:
      cand_inds = np.where(anchor_labels == 0)[0]
    else:
      cand_inds = np.where(anchor_labels >= 0)[0]
    cand_anchors = anchors[cand_inds]
    if use_masks_for_ignore:
      assert crowd_masks is not None, "get_anchor_labels: Need to supply masks!!"
      assert use_ioa_for_ignore, "get_anchor_labels: Using masks also need to use IOA"
      assert crowd_masks.shape[2] == len(crowd_boxes), "Inconsistent number of crowd boxes/masks"
      if not crowd_masks.shape[2] == 1:
        #print("Expect only one crowd mask per image!!! Merging them together now")
        crowd_masks = np.expand_dims(np.sum(crowd_masks, axis=-1), axis=-1)
      ious = calculate_ioas_using_ignore_mask(cand_anchors, crowd_masks[:, :, 0])
    else:
      if use_ioa_for_ignore:
        ious = calculate_ioas(cand_anchors, crowd_boxes)
      else:
        ious = calculate_ious(cand_anchors, crowd_boxes)
    overlap_with_crowd = cand_inds[ious.max(axis=1) > CROWD_OVERLAP_THRES]
    anchor_labels[overlap_with_crowd] = -1

  # Filter fg labels: ignore some fg if fg is too many
  target_num_fg = int(RPN_BATCH_PER_IM * RPN_FG_RATIO)
  fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
  # Note that fg could be fewer than the target ratio

  # filter bg labels. num_bg is not allowed to be too many
  old_num_bg = np.sum(anchor_labels == 0)
  if old_num_bg == 0 or len(fg_inds) == 0:
    # No valid bg/fg in this image, skip.
    # This can happen if, e.g. the image has large crowd.
    anchor_labels = np.ones((NA,), dtype='int32')
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    return anchor_labels, anchor_boxes, True
  target_num_bg = RPN_BATCH_PER_IM - len(fg_inds)
  filter_box_label(anchor_labels, 0, target_num_bg)  # ignore return values

  # Set anchor boxes: the best gt_box for each fg anchor
  anchor_boxes = np.zeros((NA, 4), dtype='float32')
  fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
  anchor_boxes[fg_inds, :] = fg_boxes
  return anchor_labels, anchor_boxes, False


def get_rpn_anchor_input(boxes, boxes_ids, is_crowd, H, W, masks, prefer_gt_to_ignore=False, use_ioa_for_ignore=False,
                         use_masks_for_ignore=False):
  """
  Args:
      boxes: nx4, floatbox, gt. shoudn't be changed
      is_crowd: n,
      H, W: size of image

  Returns:
      The anchor labels and target boxes for each pixel in the featuremap.
      fm_labels: fHxfWxNA
      fm_boxes: fHxfWxNAx4
  """
  boxes = boxes[boxes_ids > 0]
  is_crowd = is_crowd[boxes_ids > 0]

  def filter_box_inside(input):
    indices = np.where(
      (input[:, 0] >= 0) &
      (input[:, 1] >= 0) &
      (input[:, 2] <= W) &
      (input[:, 3] <= H))[0]
    return indices

  crowd_boxes = boxes[is_crowd == 1]
  non_crowd_boxes = boxes[is_crowd == 0]
  if use_masks_for_ignore:
    masks = masks[:, :, boxes_ids > 0]
    crowd_masks = masks[:, :, is_crowd == 1]
  else:
    crowd_masks = None

  # fHxfWxAx4
  featuremap_anchors_flatten = np.copy(ALL_ANCHORS).reshape((-1, 4))
  # only use anchors inside the image
  inside_ind = filter_box_inside(featuremap_anchors_flatten)
  inside_anchors = featuremap_anchors_flatten[inside_ind, :]

  anchor_labels, anchor_boxes, invalid_bgfg = get_anchor_labels(inside_anchors, non_crowd_boxes, crowd_boxes,
                                                                prefer_gt_to_ignore, use_ioa_for_ignore,
                                                                use_masks_for_ignore, crowd_masks)

  # Fill them back to original size: fHxfWx1, fHxfWx4
  featuremap_labels = -np.ones((anchorH * anchorW * NUM_ANCHOR,), dtype='int32')
  featuremap_labels[inside_ind] = anchor_labels
  featuremap_labels = featuremap_labels.reshape((anchorH, anchorW, NUM_ANCHOR))
  featuremap_boxes = np.zeros((anchorH * anchorW * NUM_ANCHOR, 4), dtype='float32')
  featuremap_boxes[inside_ind, :] = anchor_boxes
  featuremap_boxes = featuremap_boxes.reshape((anchorH, anchorW, NUM_ANCHOR, 4))
  return featuremap_labels, featuremap_boxes, invalid_bgfg


def add_rpn_data(return_dict, prefer_gt_to_ignore=False, use_ioa_for_ignore=False, use_masks_for_ignore=False):
  gt_boxes = tf.split(return_dict[DataKeys.BBOXES_y0x0y1x1], num_or_size_splits=4, axis=1)
  gt_boxes_xyxy = tf.concat([gt_boxes[1], gt_boxes[0], gt_boxes[3], gt_boxes[2]], axis=1)
  return_dict[DataKeys.BBOXES_x0y0x1y1] = gt_boxes_xyxy
  del return_dict[DataKeys.BBOXES_y0x0y1x1]
  img_shape = tf.shape(return_dict[DataKeys.IMAGES])
  get_rpn_anchor_input_part = partial(get_rpn_anchor_input, prefer_gt_to_ignore=prefer_gt_to_ignore,
                                      use_ioa_for_ignore=use_ioa_for_ignore, use_masks_for_ignore=use_masks_for_ignore)
  if use_masks_for_ignore:
    masks = return_dict[DataKeys.SEGMENTATION_MASK]
  else:
    masks = 0
  fm_labels, fm_boxes, invalid_bgfg = tf.py_func(get_rpn_anchor_input_part, [gt_boxes_xyxy, return_dict[DataKeys.IDS],
                                                 return_dict[DataKeys.IS_CROWD], img_shape[0], img_shape[1], masks],
                                                 [tf.int32, tf.float32, tf.bool], name="get_rpn_anchor_input")
  fm_labels.set_shape((anchorH, anchorW, NUM_ANCHOR))
  fm_boxes.set_shape((anchorH, anchorW, NUM_ANCHOR, 4))
  invalid_bgfg.set_shape(())
  return_dict[DataKeys.FEATUREMAP_BOXES] = fm_boxes
  return_dict[DataKeys.FEATUREMAP_LABELS] = fm_labels
  return_dict[DataKeys.SKIP_EXAMPLE] = invalid_bgfg

  return return_dict


def load_instance_seg_annotation_np(ann_filename, n_max_detections, class_ids_with_instances, id_divisor, crowd_id):
  if not type(ann_filename) == str:
    ann_filename = ann_filename.decode("utf-8")
  ann = np.array(Image.open(ann_filename))

  return load_instance_seg_annotation_from_img_array_np(ann, n_max_detections, class_ids_with_instances, id_divisor, crowd_id)


def load_instance_seg_annotation_from_img_array_np(ann, n_max_detections, class_ids_with_instances, id_divisor, crowd_id):
  obj_ids = np.unique(ann)
  obj_ids = np.array([x for x in obj_ids if x // id_divisor in class_ids_with_instances or x // id_divisor == crowd_id],
                     dtype="int32")

  assert obj_ids.size <= n_max_detections, obj_ids.size

  # they need to be padded to n_max_detections
  bboxes = np.zeros((n_max_detections, 4), dtype="float32")
  ids = np.zeros(n_max_detections, dtype="int32")
  classes = np.zeros(n_max_detections, dtype="int32")
  is_crowd = np.zeros(n_max_detections, dtype="int32")
  img_h, img_w = ann.shape[:2]
  masks = np.zeros((img_h, img_w, n_max_detections), dtype="uint8")

  for idx, obj_id in enumerate(obj_ids):
    mask = (ann == obj_id).astype("uint8")
    masks[:, :, idx] = mask
    bbox = get_bbox_from_segmentation_mask_np(mask)
    bboxes[idx] = bbox
    #ids[idx] = (obj_id % id_divisor) + 1
    # we keep the class info in the id
    cat_id = obj_id // id_divisor
    if cat_id == crowd_id:
      is_crowd[idx] = 1
      ids[idx] = crowd_id * id_divisor
    else:
      classes[idx] = class_ids_with_instances.index(cat_id) + 1
      ids[idx] = obj_id + 1

  return bboxes, ids, classes, is_crowd, masks
