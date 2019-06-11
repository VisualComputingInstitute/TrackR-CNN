import tensorflow as tf

# Stuff copied over from https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/, small changes (NHWC)
TRAIN_PRE_NMS_TOPK = 12000
TRAIN_POST_NMS_TOPK = 2000
TEST_PRE_NMS_TOPK = 6000
TEST_POST_NMS_TOPK = 1000

RPN_MIN_SIZE = 0
RPN_PROPOSAL_NMS_THRESH = 0.7

FASTRCNN_FG_THRESH = 0.5
# fg ratio in a ROI batch
FASTRCNN_FG_RATIO = 0.25

RESULTS_PER_IM = 100

FASTRCNN_NMS_THRESH = 0.5


def decode_bbox_target(BBOX_DECODE_CLIP, box_predictions, anchors):
  """
  Args:
      box_predictions: (..., 4), logits
      anchors: (..., 4), floatbox. Must have the same shape
  Returns:
      box_decoded: (..., 4), float32. With the same shape.
  """
  orig_shape = tf.shape(anchors)
  box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
  box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
  # each is (...)x1x2
  anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
  anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

  waha = anchors_x2y2 - anchors_x1y1
  xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

  wbhb = tf.exp(tf.minimum(
    box_pred_twth, BBOX_DECODE_CLIP)) * waha
  xbyb = box_pred_txty * waha + xaya
  x1y1 = xbyb - wbhb * 0.5
  x2y2 = xbyb + wbhb * 0.5  # (...)x1x2
  out = tf.concat([x1y1, x2y2], axis=-2)
  return tf.reshape(out, orig_shape)


def encode_bbox_target(boxes, anchors):
  """
  Args:
      boxes: (..., 4), float32
      anchors: (..., 4), float32
  Returns:
      box_encoded: (..., 4), float32 with the same shape.
  """
  anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
  anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
  waha = anchors_x2y2 - anchors_x1y1
  xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

  boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
  boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
  wbhb = boxes_x2y2 - boxes_x1y1
  xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

  # Note that here not all boxes are valid. Some may be zero
  txty = (xbyb - xaya) / waha
  twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
  encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
  return tf.reshape(encoded, tf.shape(boxes))


def clip_boxes(boxes, window, name=None):
  """
  Args:
      boxes: nx4, xyxy
      window: [h, w]
  """
  boxes = tf.maximum(boxes, 0.0)
  m = tf.tile(tf.reverse(window, [0]), [2])  # (4,)
  boxes = tf.minimum(boxes, tf.to_float(m), name=name)
  return boxes


def generate_rpn_proposals(boxes, scores, img_shape, is_training):
  """
  Args:
      boxes: nx4 float dtype, decoded to floatbox already
      scores: n float, the logits
      img_shape: [h, w]
  Returns:
      boxes: kx4 float
      scores: k logits
  """
  assert boxes.shape.ndims == 2, boxes.shape
  if is_training:
    PRE_NMS_TOPK = TRAIN_PRE_NMS_TOPK
    POST_NMS_TOPK = TRAIN_POST_NMS_TOPK
  else:
    PRE_NMS_TOPK = TEST_PRE_NMS_TOPK
    POST_NMS_TOPK = TEST_POST_NMS_TOPK

  topk = tf.minimum(PRE_NMS_TOPK, tf.size(scores))
  topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
  topk_boxes = tf.gather(boxes, topk_indices)
  topk_boxes = clip_boxes(topk_boxes, img_shape)

  topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
  topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
  # nx1x2 each
  wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
  valid = tf.reduce_all(wbhb > RPN_MIN_SIZE, axis=1)  # n,
  topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
  topk_valid_scores = tf.boolean_mask(topk_scores, valid)

  topk_valid_boxes_y1x1y2x2 = tf.reshape(
    tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
    (-1, 4), name='nms_input_boxes')
  # Version from tensorflow 1.10, where everything works
  from tensorflow.python.ops import gen_image_ops
  tf.image.non_max_suppression = gen_image_ops.non_max_suppression
  nms_indices = tf.image.non_max_suppression(
    topk_valid_boxes_y1x1y2x2,
    topk_valid_scores,
    max_output_size=POST_NMS_TOPK,
    iou_threshold=RPN_PROPOSAL_NMS_THRESH)

  topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
  final_boxes = tf.gather(
    topk_valid_boxes,
    nms_indices, name='boxes')
  final_scores = tf.gather(topk_valid_scores, nms_indices, name='scores')
  tf.sigmoid(final_scores, name='probs')  # for visualization
  return final_boxes, final_scores


def area(boxes):
    """
    Args:
      boxes: nx4 floatbox
    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def pairwise_intersection(boxlist1, boxlist2):
  """Compute pairwise intersection areas between boxes.
  Args:
    boxlist1: Nx4 floatbox
    boxlist2: Mx4
  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
  x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
  all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
  all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
  intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
  all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
  intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def pairwise_iou(boxlist1, boxlist2):
  """Computes pairwise intersection-over-union between box collections.
  Args:
    boxlist1: Nx4 floatbox
    boxlist2: Mx4
  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  intersections = pairwise_intersection(boxlist1, boxlist2)
  areas1 = area(boxlist1)
  areas2 = area(boxlist2)
  unions = (
      tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
  return tf.where(
    tf.equal(intersections, 0.0),
    tf.zeros_like(intersections), tf.truediv(intersections, unions))


def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels, fastrcnn_batch_per_img, target_ids):
  """
  Sample some ROIs from all proposals for training.
  Args:
      boxes: nx4 region proposals, floatbox
      gt_boxes: mx4, floatbox
      gt_labels: m, int32
  Returns:
      sampled_boxes: tx4 floatbox, the rois
      sampled_labels: t labels, in [0, #class-1]. Positive means foreground.
      fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
          It contains the matching GT of each foreground roi.
  """
  iou = pairwise_iou(boxes, gt_boxes)  # nxm
  # proposal_metrics(iou)

  # add ground truth as proposals as well
  boxes = tf.concat([boxes, gt_boxes], axis=0)  # (n+m) x 4
  iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # (n+m) x m

  # #proposal=n+m from now on

  def sample_fg_bg(iou):
    fg_mask = tf.reduce_max(iou, axis=1) >= FASTRCNN_FG_THRESH

    fg_inds = tf.reshape(tf.where(fg_mask), [-1])
    num_fg = tf.minimum(int(
      fastrcnn_batch_per_img * FASTRCNN_FG_RATIO),
      tf.size(fg_inds), name='num_fg')
    fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

    bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
    num_bg = tf.minimum(
      fastrcnn_batch_per_img - num_fg,
      tf.size(bg_inds), name='num_bg')
    bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

    # add_moving_summary(num_fg, num_bg)
    return fg_inds, bg_inds

  fg_inds, bg_inds = sample_fg_bg(iou)
  # fg,bg indices w.r.t proposals

  best_iou_ind = tf.argmax(iou, axis=1)  # #proposal, each in 0~m-1
  fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)  # num_fg

  all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # indices w.r.t all n+m proposal boxes
  ret_boxes = tf.gather(boxes, all_indices, name='sampled_proposal_boxes')

  ret_labels = tf.concat(
    [tf.gather(gt_labels, fg_inds_wrt_gt),
     tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name='sampled_labels')

  ret_target_ids = tf.concat([tf.gather(target_ids, fg_inds_wrt_gt), tf.zeros_like(bg_inds, dtype=target_ids.dtype)],
                             axis=0, name="sampled_target_ids")
  # stop the gradient -- they are meant to be ground-truth
  return tf.stop_gradient(ret_boxes), tf.stop_gradient(ret_labels), fg_inds_wrt_gt, tf.stop_gradient(ret_target_ids)


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
  """
  Better-aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
  Args:
      image: NHWC
      boxes: nx4, x1y1x2y2
      box_ind: (n,)
      crop_size (int):
  Returns:
      n,size,size,C
  """
  assert isinstance(crop_size, int), crop_size

  # TF's crop_and_resize produces zeros on border
  if pad_border:
    # this can be quite slow
    image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
    boxes = boxes + 1

  def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
    """
    The way tf.image.crop_and_resize works (with normalized box):
    Initial point (the value of output[0]): x0_box * (W_img - 1)
    Spacing: w_box * (W_img - 1) / (W_crop - 1)
    Use the above grid to bilinear sample.
    However, what we want is (with fpcoor box):
    Spacing: w_box / W_crop
    Initial point: x0_box + spacing/2 - 0.5
    (-0.5 because bilinear sample assumes floating point coordinate (0.0, 0.0) is the same as pixel value (0, 0))
    This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
    Returns:
        y1x1y2x2
    """
    x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

    spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
    spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

    nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
    ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

    nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
    nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

    return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

  image_shape = tf.shape(image)[1:3]
  boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
  ret = tf.image.crop_and_resize(
    image, boxes, box_ind,
    crop_size=[crop_size, crop_size])
  return ret


def roi_align(featuremap, boxes, output_shape):
  """
  Args:
      featuremap: 1xHxWxC
      boxes: Nx4 floatbox
      output_shape: int
  Returns:
      NxoHxoWxC
  """
  boxes = tf.stop_gradient(boxes)  # TODO
  # sample 4 locations per roi bin
  ret = crop_and_resize(
    featuremap, boxes,
    tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
    output_shape * 2)
  ret = tf.nn.avg_pool(ret, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC')
  return ret


def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
  """
  Args:
      anchor_labels: fHxfWxNA
      anchor_boxes: fHxfWxNAx4, encoded
      label_logits:  fHxfWxNA
      box_logits: fHxfWxNAx4
  Returns:
      label_loss, box_loss
  """
  with tf.device('/cpu:0'):
    valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
    pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
    nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
    nr_pos = tf.count_nonzero(pos_mask, dtype=tf.int32, name='num_pos_anchor')

    valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
  valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

  with tf.name_scope('label_metrics'):
    valid_label_prob = tf.nn.sigmoid(valid_label_logits)
    summaries = []
    with tf.device('/cpu:0'):
      for th in [0.5, 0.2, 0.1]:
        valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
        nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
        pos_prediction_corr = tf.count_nonzero(
          tf.logical_and(
            valid_label_prob > th,
            tf.equal(valid_prediction, valid_anchor_labels)),
          dtype=tf.int32)
        summaries.append(tf.truediv(
          pos_prediction_corr,
          nr_pos, name='recall_th{}'.format(th)))
        precision = tf.to_float(tf.truediv(pos_prediction_corr, nr_pos_prediction))
        precision = tf.where(tf.equal(nr_pos_prediction, 0), 0.0, precision, name='precision_th{}'.format(th))
        summaries.append(precision)
    # add_moving_summary(*summaries)

  label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.to_float(valid_anchor_labels), logits=valid_label_logits)
  label_loss = tf.reduce_mean(label_loss, name='label_loss')

  pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
  pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
  delta = 1.0 / 9
  box_loss = tf.losses.huber_loss(
    pos_anchor_boxes, pos_box_logits, delta=delta,
    reduction=tf.losses.Reduction.SUM) / delta
  box_loss = tf.div(
    box_loss,
    tf.cast(nr_valid, tf.float32), name='box_loss')

  # add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
  return label_loss, box_loss


def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
  """
  Args:
      labels: n,
      label_logits: nxC
      fg_boxes: nfgx4, encoded
      fg_box_logits: nfgx(C-1)x4
  """
  label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=label_logits)
  label_loss = tf.reduce_mean(label_loss, name='label_loss')

  fg_inds = tf.where(labels > 0)[:, 0]
  fg_labels = tf.gather(labels, fg_inds)
  num_fg = tf.size(fg_inds)
  indices = tf.stack(
    [tf.range(num_fg),
     tf.to_int32(fg_labels) - 1], axis=1)  # #fgx2
  fg_box_logits = tf.gather_nd(fg_box_logits, indices)

  # with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
  #   prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
  #   correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
  #   accuracy = tf.reduce_mean(correct, name='accuracy')
  #   fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
  #   num_zero = tf.reduce_sum(tf.to_int32(tf.equal(fg_label_pred, 0)), name='num_zero')
  #   false_negative = tf.truediv(num_zero, num_fg, name='false_negative')
  #   fg_accuracy = tf.reduce_mean(
  #     tf.gather(correct, fg_inds), name='fg_accuracy')

  box_loss = tf.losses.huber_loss(
    fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
  box_loss = tf.truediv(
    box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

  # add_moving_summary(label_loss, box_loss, accuracy, fg_accuracy, false_negative)
  return label_loss, box_loss


def fastrcnn_predictions(boxes, probs, num_classes, result_score_thresh, class_agnostic_boxes=False):
  """
  Generate final results from predictions of all proposals.
  Args:
      boxes: n#catx4 floatbox in float32
      probs: nx#class
  """
  if class_agnostic_boxes:
    assert boxes.shape[1] == 1, (boxes.shape[1], 1)
  else:
    assert boxes.shape[1] == num_classes - 1, (boxes.shape[1], num_classes - 1)
  assert probs.shape[1] == num_classes, (probs.shape[1], num_classes)
  boxes = tf.transpose(boxes, [1, 0, 2])  # #catxnx4
  probs = tf.transpose(probs[:, 1:], [1, 0])  # #catxn

  if class_agnostic_boxes:
    num_cat = tf.shape(probs)[0]
    boxes = tf.tile(boxes, [num_cat, 1, 1])

  def f(X):
    """
    prob: n probabilities
    box: nx4 boxes
    Returns: n boolean, the selection
    """
    prob, box = X
    output_shape = tf.shape(prob)
    # filter by score threshold
    ids = tf.reshape(tf.where(prob > result_score_thresh), [-1])
    prob = tf.gather(prob, ids)
    box = tf.gather(box, ids)
    # NMS within each class
    # Version from tensorflow 1.10, where everything works
    from tensorflow.python.ops import gen_image_ops
    tf.image.non_max_suppression = gen_image_ops.non_max_suppression  # or v3, v4
    selection = tf.image.non_max_suppression(
      box, prob, RESULTS_PER_IM, FASTRCNN_NMS_THRESH)
    selection = tf.to_int32(tf.gather(ids, selection))
    # sort available in TF>1.4.0
    # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
    sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
    mask = tf.sparse_to_dense(
      sparse_indices=sorted_selection,
      output_shape=output_shape,
      sparse_values=True,
      default_value=False)
    return mask

  masks = tf.map_fn(f, (probs, boxes), dtype=tf.bool,
                    parallel_iterations=10)  # #cat x N
  selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
  probs = tf.boolean_mask(probs, masks)

  # filter again by sorting scores
  topk_probs, topk_indices = tf.nn.top_k(
    probs,
    tf.minimum(RESULTS_PER_IM, tf.size(probs)),
    sorted=False)
  filtered_selection = tf.gather(selected_indices, topk_indices)
  filtered_selection = tf.reverse(filtered_selection, axis=[1], name='filtered_indices')
  return filtered_selection, topk_probs


def maskrcnn_loss(mask_logits, fg_labels, fg_target_masks, class_agnostic):
  """
  Args:
      mask_logits: #fg x #category x14x14
      fg_labels: #fg, in 1~#class
      fg_target_masks: #fgx14x14, int
  """
  if class_agnostic:
    # probably we could also set this to tf.ones?
    fg_labels = tf.cast(tf.greater(fg_labels, 0), fg_labels.dtype)
  num_fg = tf.size(fg_labels)
  indices = tf.stack([tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)  # #fgx2
  # mask_logits needs to be NCHW for the next operation
  mask_logits = tf.transpose(mask_logits, [0, 3, 1, 2])
  mask_logits = tf.gather_nd(mask_logits, indices)  # #fgx14x14
  mask_probs = tf.sigmoid(mask_logits)

  # add some training visualizations to tensorboard
  with tf.name_scope('mask_viz'):
    viz = tf.concat([fg_target_masks, mask_probs], axis=1)
    viz = tf.expand_dims(viz, 3)
    viz = tf.cast(viz * 255, tf.uint8, name='viz')
    tf.summary.image('mask_truth|pred', viz, max_outputs=10)

  loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=fg_target_masks, logits=mask_logits)
  loss = tf.reduce_mean(loss, name='maskrcnn_loss')

  pred_label = mask_probs > 0.5
  truth_label = fg_target_masks > 0.5
  accuracy = tf.reduce_mean(
    tf.to_float(tf.equal(pred_label, truth_label)),
    name='accuracy')
  pos_accuracy = tf.logical_and(
    tf.equal(pred_label, truth_label),
    tf.equal(truth_label, True))
  pos_accuracy = tf.reduce_mean(tf.to_float(pos_accuracy), name='pos_accuracy')
  fg_pixel_ratio = tf.reduce_mean(tf.to_float(truth_label), name='fg_pixel_ratio')

  # add_moving_summary(loss, accuracy, fg_pixel_ratio, pos_accuracy)
  return loss
