# Cf. https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/train.py
# and https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/model.py

import tensorflow as tf
import numpy as np
from functools import partial

from datasets import DataKeys
from network.ConvolutionalLayers import Conv, ConvTranspose
from network.FullyConnected import FullyConnected
from network.Layer import Layer
from network.Resnet import add_resnet_conv5
from core import Measures, Extractions
from network.FasterRCNN_utils import decode_bbox_target, encode_bbox_target,\
  generate_rpn_proposals, sample_fast_rcnn_targets, roi_align, rpn_losses,\
  fastrcnn_losses, clip_boxes, fastrcnn_predictions, maskrcnn_loss, crop_and_resize
from datasets.util.Detection import ALL_ANCHORS, NUM_ANCHOR, ANCHOR_STRIDE


FASTRCNN_BBOX_REG_WEIGHTS = np.array([10, 10, 5, 5], dtype='float32')


def rpn_head(featuremap, channel, num_anchors, tower_setup):
  with tf.variable_scope('rpn'):
    hidden = Conv('conv0', [featuremap], channel, tower_setup, old_order=True, bias=True,
                  W_initializer=tf.random_normal_initializer(stddev=0.01)).outputs[0]

    label_logits = Conv('class', [hidden], num_anchors, tower_setup, (1, 1), old_order=True, bias=True,
                        activation="linear", W_initializer=tf.random_normal_initializer(stddev=0.01)).outputs[0]
    box_logits = Conv('box', [hidden], 4 * num_anchors, tower_setup, (1, 1), old_order=True, bias=True,
                      activation="linear", W_initializer=tf.random_normal_initializer(stddev=0.01)).outputs[0]
    shp = tf.shape(box_logits)
    box_logits = tf.reshape(box_logits, tf.stack([shp[0], shp[1], shp[2], num_anchors, 4]))
  return label_logits, box_logits


def fastrcnn_head(feature, num_classes, reid_dim, tower_setup, reid_per_class=False, class_agnostic_box=False):
  with tf.variable_scope('fastrcnn'):
    # GlobalAvgPooling, see https://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/models/pool.html
    assert feature.shape.ndims == 4
    feature = tf.reduce_mean(feature, [1, 2], name='gap/output')

    classification = FullyConnected("class", [feature], num_classes, tower_setup, activation="linear",
                                    W_initializer=tf.random_normal_initializer(stddev=0.01)).outputs[0]
    if class_agnostic_box:
      num_hidden_box = 1
    else:
      num_hidden_box = num_classes - 1
    box_regression = FullyConnected("box", [feature], num_hidden_box * 4, tower_setup,
                                    activation="linear",
                                    W_initializer=tf.random_normal_initializer(stddev=0.001)).outputs[0]
    box_regression = tf.reshape(box_regression, (-1, num_hidden_box, 4))

    if reid_dim > 0:
      if reid_per_class:
        reid_features = \
          FullyConnected("reid", [feature], (num_classes - 1) * reid_dim, tower_setup, activation="linear").outputs[0]
        reid_features = tf.reshape(reid_features, (-1, (num_classes - 1), reid_dim))
        best_class = tf.cast(tf.argmax(classification, axis=-1) - 1, dtype=tf.int32)
        feature_indices = tf.stack([tf.range(tf.shape(best_class)[0], dtype=tf.int32), best_class], axis=1)
        reid_features = tf.gather_nd(reid_features, feature_indices)
      else:
        reid_features = FullyConnected("reid", [feature], reid_dim, tower_setup, activation="linear").outputs[0]
    else:
      reid_features = None
    return classification, box_regression, reid_features


def maskrcnn_head(feature, num_class, tower_setup, class_agnostic_conv=False):
  with tf.variable_scope('maskrcnn'):
    # c2's MSRAFill is fan_out
    l = ConvTranspose('deconv', [feature], 256, tower_setup, (2, 2), strides=(2, 2), bias=True,
                      W_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out',
                                                                    distribution='truncated_normal')).outputs[0]
    if class_agnostic_conv:
      num_output_channels = 1
    else:
      num_output_channels = num_class - 1
    l = Conv('conv', [l], num_output_channels, tower_setup, (1, 1), old_order=True, bias=True, activation="linear",
             W_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out',
                                                           distribution='truncated_normal')).outputs[0]
  return l


# See also https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/train.py
class FasterRCNN(Layer):
  def _get_anchors(self, shape2d):
    """
    Returns:
        FSxFSxNAx4 anchors,
    """
    # FSxFSxNAx4 (FS=MAX_SIZE//ANCHOR_STRIDE)
    with tf.name_scope('anchors'):
      all_anchors = tf.constant(ALL_ANCHORS, name='all_anchors', dtype=tf.float32)
      fm_anchors = tf.slice(
        all_anchors, [0, 0, 0, 0], tf.stack([
          shape2d[0], shape2d[1], -1, -1]), name='fm_anchors')
    return fm_anchors

  @staticmethod
  def fill_full_mask(boxes, masks, img_shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    n_boxes = boxes.shape[0]
    assert n_boxes == masks.shape[0]
    ret = np.zeros([n_boxes, img_shape[0], img_shape[1]], dtype='uint8')
    for idx in range(n_boxes):
      # int() is floor
      # box fpcoor=0.0 -> intcoor=0.0
      x0, y0 = list(map(int, boxes[idx, :2] + 0.5))
      # box fpcoor=h -> intcoor=h-1, inclusive
      x1, y1 = list(map(int, boxes[idx, 2:] - 0.5))  # inclusive
      x1 = max(x0, x1)  # require at least 1x1
      y1 = max(y0, y1)

      w = x1 + 1 - x0
      h = y1 + 1 - y0

      # rounding errors could happen here, because masks were not originally computed for this shape.
      # but it's hard to do better, because the network does not know the "original" scale
      import cv2
      mask = (cv2.resize(masks[idx, :, :], (w, h)) > 0.5).astype('uint8')
      ret[idx, y0:y1 + 1, x0:x1 + 1] = mask

    return ret

  def __init__(self, name, inputs, network_input_dict, tower_setup, fastrcnn_batch_per_img=256,
               result_score_thresh=0.05, reid_dimension=0, reid_per_class=False, reid_loss_per_class=False,
               reid_loss_worst_examples_percent=1.0, reid_loss_variant=0, reid_loss_factor=1.0,
               reid_measure="cosine", reid_adjacent_frames=False, class_agnostic_box_and_mask_heads=False,
               reid_loss_margin=0.2, provide_boxes_as_input=False):
    super(FasterRCNN, self).__init__()
    self.is_training = tower_setup.is_training
    inputs = inputs[0]
    if self.is_training:
      tf.add_to_collection('checkpoints', inputs)
    self._image_shape2d = tf.shape(network_input_dict[DataKeys.IMAGES])[1:3]
    self._add_maskrcnn = tower_setup.dataset.config.bool("add_masks", True)
    max_size = tower_setup.dataset.config.int_list("input_size_train", [])[1]
    self.bbox_decode_clip = np.log(max_size / 16.0)
    self.fastrcnn_batch_per_img = fastrcnn_batch_per_img
    self.result_score_thresh = result_score_thresh
    self.tower_setup = tower_setup
    self.network_input_dict = network_input_dict
    self._num_classes = tower_setup.dataset.num_classes()
    self._do_reid = reid_dimension > 0
    self._reid_dimension = reid_dimension
    self._reid_per_class = reid_per_class
    self._reid_loss_per_class = reid_loss_per_class
    self._reid_loss_worst_examples_percent = reid_loss_worst_examples_percent
    # Loss variants: 0=sig_ce - NO LONGER IMPLEMENTED, 1=batch_hard, 2=batch_all, 3=batch_all_no_zeros, 4=contrastive
    self._reid_loss_variant = reid_loss_variant
    self._reid_loss_factor = reid_loss_factor
    # ReID distance: cosine, euclidean, normalized_euclidean
    self._reid_measure = reid_measure
    self._reid_adjacent_frames = reid_adjacent_frames
    self._reid_loss_margin = reid_loss_margin
    self.class_agnostic_box_and_mask_heads = class_agnostic_box_and_mask_heads
    self.provide_boxes_as_input = provide_boxes_as_input
    self.network_input_dict = network_input_dict

    with tf.variable_scope(name):
      rpn_label_logits, rpn_box_logits = rpn_head(inputs, 1024, NUM_ANCHOR, self.tower_setup)
      fm_shape = tf.shape(inputs)[1:3]  # h,w
      fm_anchors = self._get_anchors(fm_shape)

    losses = []
    reid_features_and_target_ids_per_time = []
    batch_size = inputs.get_shape().as_list()[0]
    assert batch_size is not None
    # Prepare outputs
    self.outputs_per_batch_idx = [[] for _ in range(batch_size)]
    self.extractions_per_batch_idx = [{} for _ in range(batch_size)]
    final_boxes_list = []
    for batch_idx in range(batch_size):
      with tf.control_dependencies(control_inputs=final_boxes_list):
        with tf.variable_scope(name, reuse=True if batch_idx > 0 else None):
          final_boxes, final_labels, final_masks, final_probs, reid_features, target_ids, final_reid_features, \
            fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss = \
            self._create_heads(batch_idx, fm_anchors, fm_shape, rpn_box_logits, inputs, rpn_label_logits)
          reid_features_and_target_ids_per_time.append((reid_features, target_ids))
          if self.is_training:
            losses.extend([fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss])
            # combine individual losses for summaries create summaries (atm they are separate)
            self.add_scalar_summary(rpn_label_loss, "rpn_label_loss")
            self.add_scalar_summary(rpn_box_loss, "rpn_box_loss")
            self.add_scalar_summary(fastrcnn_label_loss, "fastrcnn_label_loss")
            self.add_scalar_summary(fastrcnn_box_loss, "fastrcnn_box_loss")
          else:
            self._add_test_measures(batch_idx, final_boxes, final_probs, final_labels, final_masks, final_reid_features)
            final_boxes_list.append(final_boxes)  # For training, didnt have any OOM errors

    # Sanity check
    #for outputs in self.outputs_per_batch_idx:
    #  assert len(outputs) == len(self.outputs_per_batch_idx[0])
    #for ext in self.extractions_per_batch_idx:
    #  assert ext.keys() == self.extractions_per_batch_idx[0].keys()
    #num_outputs = len(self.outputs_per_batch_idx[0])
    # This doesn't work due to different sized outputs
    #for i in range(num_outputs):
    #  self.outputs.append(tf.stack([outputs[i] for outputs in self.outputs_per_batch_idx]))
    self.outputs = self.outputs_per_batch_idx[0]
    for key in self.extractions_per_batch_idx[0].keys():
      # To stack the extraction results
      #self.extractions[key] = tf.stack([extractions[key] for extractions in self.extractions_per_batch_idx])
      self.extractions[key] = [extractions[key] for extractions in self.extractions_per_batch_idx]

    if self.is_training and self._do_reid and self._reid_loss_factor > 0.0:
      reid_loss = create_reid_loss(reid_features_and_target_ids_per_time, self._reid_loss_per_class,
                                   self._reid_loss_worst_examples_percent, self._reid_loss_variant,
                                   self._reid_loss_factor, self._reid_measure, self._reid_adjacent_frames,
                                   self._reid_loss_margin)
      losses.append(reid_loss)
      self.add_scalar_summary(reid_loss, "reid_loss")

    if self.is_training:
      vars_to_regularize = tf.trainable_variables("frcnn/(?:rpn|group3|fastrcnn|maskrcnn)/.*W")
      regularizers = [1e-4 * tf.nn.l2_loss(W) for W in vars_to_regularize]
      regularization_loss = tf.add_n(regularizers, "regularization_loss")
      self.regularizers.append(regularization_loss)
      # TODO how to properly weight losses?
      loss = tf.add_n(losses, 'total_cost') / batch_size
      self.losses.append(loss)
      # self.add_scalar_summary(regularization_loss, "regularization_loss")
    else:
      loss = 0.0
    self.add_scalar_summary(loss, "loss")
    self._add_basic_measures(inputs, loss)

  def _create_heads(self, batch_idx, fm_anchors, fm_shape, rpn_box_logits, rpn_input, rpn_label_logits):
    rpn_label_logits = rpn_label_logits[batch_idx]
    rpn_box_logits = rpn_box_logits[batch_idx]
    rpn_input = rpn_input[batch_idx, tf.newaxis]

    decoded_boxes = decode_bbox_target(self.bbox_decode_clip, rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
    proposal_boxes, proposal_scores = generate_rpn_proposals(
      tf.reshape(decoded_boxes, [-1, 4]),
      tf.reshape(rpn_label_logits, [-1]),
      self._image_shape2d, self.is_training)

    if self.provide_boxes_as_input:
      input_boxes = tf.squeeze(self.network_input_dict[DataKeys.BBOXES_TO_REFINE_x0y0x1y1], axis=0)
      image_shape2d_before_resize = self.network_input_dict[DataKeys.RAW_IMAGE_SIZES]
      assert image_shape2d_before_resize.shape[0] == 1, "we assume batch size is 1 for now"
      image_shape2d_before_resize = tf.squeeze(image_shape2d_before_resize, axis=0)
      image_shape2d = self._image_shape2d
      old_height, old_width = image_shape2d_before_resize[0], image_shape2d_before_resize[1]
      new_height, new_width = image_shape2d[0], image_shape2d[1]
      height_scale = new_height / old_height
      width_scale = new_width / old_width
      scale = tf.stack([width_scale, height_scale, width_scale, height_scale], axis=0)
      proposal_boxes = input_boxes * tf.cast(scale, tf.float32)

    if self.is_training:
      # Prepare the data, slice inputs by batch index
      gt_boxes = self.network_input_dict[DataKeys.BBOXES_x0y0x1y1]
      gt_boxes = gt_boxes[batch_idx]
      gt_labels = self.network_input_dict[DataKeys.CLASSES]
      gt_labels = tf.cast(gt_labels[batch_idx], dtype=tf.int64)
      if self._add_maskrcnn:
        gt_masks = self.network_input_dict[DataKeys.SEGMENTATION_MASK]
        gt_masks = gt_masks[batch_idx]
        gt_masks = tf.transpose(gt_masks, [2, 0, 1])
      else:
        gt_masks = None
      is_crowd = self.network_input_dict[DataKeys.IS_CROWD]
      is_crowd = is_crowd[batch_idx]
      # gt_crowd_boxes = tf.boolean_mask(gt_boxes, is_crowd)
      target_ids = self.network_input_dict[DataKeys.IDS]  # If 0, ignore; but 1 for IS_CROWD==True
      target_ids = target_ids[batch_idx]
      valid_gts_mask = tf.logical_and(tf.greater(target_ids, 0), tf.equal(is_crowd, 0))
      gt_boxes = tf.boolean_mask(gt_boxes, valid_gts_mask)
      gt_labels = tf.boolean_mask(gt_labels, valid_gts_mask)
      target_ids = tf.boolean_mask(target_ids, valid_gts_mask)
      featuremap_labels = self.network_input_dict[DataKeys.FEATUREMAP_LABELS]
      featuremap_labels = featuremap_labels[batch_idx]
      featuremap_boxes = self.network_input_dict[DataKeys.FEATUREMAP_BOXES]  # already in xyxy format
      featuremap_boxes = featuremap_boxes[batch_idx]

      # sample proposal boxes in training
      rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, rcnn_target_ids = sample_fast_rcnn_targets(
        proposal_boxes, gt_boxes, gt_labels, self.fastrcnn_batch_per_img, target_ids)
      boxes_on_featuremap = rcnn_sampled_boxes * (1.0 / ANCHOR_STRIDE)
      # tf.add_to_collection('checkpoints', boxes_on_featuremap)
    else:
      gt_boxes, gt_labels, gt_masks, target_ids, featuremap_labels, featuremap_boxes = [None] * 6
      rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, rcnn_target_ids = None, None, None, None
      # use all proposal boxes in inference
      boxes_on_featuremap = proposal_boxes * (1.0 / ANCHOR_STRIDE)

    fastrcnn_box_logits, fastrcnn_label_logits, feature_fastrcnn, reid_features = self._create_fastrcnn_output(
      boxes_on_featuremap, rpn_input)

    if self.is_training:
      #tf.add_to_collection('checkpoints', fastrcnn_box_logits)
      #tf.add_to_collection('checkpoints', fastrcnn_label_logits)
      #tf.add_to_collection('checkpoints', feature_fastrcnn)
      fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss = self._create_losses(
        featuremap_labels, featuremap_boxes, fm_anchors, fm_shape, fastrcnn_box_logits, fastrcnn_label_logits,
        feature_fastrcnn, fg_inds_wrt_gt, gt_boxes, gt_masks, rcnn_labels, rcnn_sampled_boxes, rpn_box_logits,
        rpn_label_logits)
      final_boxes, final_labels, final_masks, final_probs, final_reid_features = None, None, None, None, None
    else:
      final_boxes, final_labels, final_masks, final_probs, final_reid_features = self._create_final_outputs(
        batch_idx, rpn_input, fastrcnn_box_logits, fastrcnn_label_logits, proposal_boxes, reid_features)
      fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss = None, None, None, None, None

    return final_boxes, final_labels, final_masks, final_probs, reid_features, rcnn_target_ids, final_reid_features, \
        fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss

  def _create_fastrcnn_output(self, boxes_on_featuremap, rpn_input):
    roi_resized = roi_align(rpn_input, boxes_on_featuremap, 14)

    def ff_true():
      feature_fastrcnn_ = add_resnet_conv5(roi_resized, self.tower_setup)[0]
      fastrcnn_label_logits_, fastrcnn_box_logits_, fastrcnn_reid_features_ = fastrcnn_head(
        feature_fastrcnn_, self._num_classes, self._reid_dimension, self.tower_setup, self._reid_per_class,
        class_agnostic_box=self.class_agnostic_box_and_mask_heads)
      if self._do_reid:
        return feature_fastrcnn_, fastrcnn_label_logits_, fastrcnn_box_logits_, fastrcnn_reid_features_
      else:
        return feature_fastrcnn_, fastrcnn_label_logits_, fastrcnn_box_logits_

    def ff_false():
      ncls = self._num_classes
      if self._do_reid:
        return tf.zeros([0, 7, 7, 2048]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4]), \
               tf.zeros([0, self._reid_dimension])
      else:
        return tf.zeros([0, 7, 7, 2048]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

    feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits, *fastrcnn_reid_features = tf.cond(
      tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)
    if len(fastrcnn_reid_features) == 0:
      fastrcnn_reid_features = None
    else:
      assert len(fastrcnn_reid_features) == 1
      fastrcnn_reid_features = fastrcnn_reid_features[0]
    return fastrcnn_box_logits, fastrcnn_label_logits, feature_fastrcnn, fastrcnn_reid_features

  def _create_losses(self, featuremap_labels, featuremap_boxes, fm_anchors, fm_shape, fastrcnn_box_logits,
                     fastrcnn_label_logits, feature_fastrcnn, fg_inds_wrt_gt, gt_boxes, gt_masks,
                     rcnn_labels, rcnn_sampled_boxes, rpn_box_logits, rpn_label_logits):
    anchor_labels = tf.slice(
      featuremap_labels, [0, 0, 0],
      tf.stack([fm_shape[0], fm_shape[1], -1]),
      name='sliced_anchor_labels')
    anchor_boxes = tf.slice(
      featuremap_boxes, [0, 0, 0, 0],
      tf.stack([fm_shape[0], fm_shape[1], -1, -1]),
      name='sliced_anchor_boxes')
    anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)

    # rpn loss
    rpn_label_loss, rpn_box_loss = rpn_losses(
      anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)
    # fastrcnn loss
    fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])  # fg inds w.r.t all samples
    fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)
    matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
    encoded_boxes = encode_bbox_target(
      matched_gt_boxes,
      fg_sampled_boxes) * tf.constant(FASTRCNN_BBOX_REG_WEIGHTS)
    fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
      rcnn_labels, fastrcnn_label_logits,
      encoded_boxes,
      tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))

    if self._add_maskrcnn:
      # maskrcnn loss
      fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
      fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
      mask_logits = maskrcnn_head(fg_feature, self._num_classes, self.tower_setup,
                                  class_agnostic_conv=self.class_agnostic_box_and_mask_heads)  # #fg x #cat x 14x14

      gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
      target_masks_for_fg = crop_and_resize(
        tf.expand_dims(gt_masks_for_fg, 3),
        fg_sampled_boxes,
        tf.range(tf.size(fg_inds_wrt_gt)), 14, pad_border=False)  # nfg x 1x14x14
      target_masks_for_fg = tf.squeeze(target_masks_for_fg, 3, 'sampled_fg_mask_targets')
      mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg,
                                 class_agnostic=self.class_agnostic_box_and_mask_heads)
      self.add_scalar_summary(mrcnn_loss, "mrcnn_loss")
    else:
      mrcnn_loss = 0.0

    return fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss

  def _create_final_outputs(self, batch_idx, rpn_input, fastrcnn_box_logits, fastrcnn_label_logits,
                            proposal_boxes, reid_features):
    label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
    if self.class_agnostic_box_and_mask_heads:
      anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, 1, 1])  # #proposal x #Cat x 4
    else:
      anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, self._num_classes - 1, 1])  # #proposal x #Cat x 4
    decoded_boxes = decode_bbox_target(
      self.bbox_decode_clip,
      fastrcnn_box_logits /
      tf.constant(FASTRCNN_BBOX_REG_WEIGHTS), anchors)
    decoded_boxes = clip_boxes(decoded_boxes, self._image_shape2d, name='fastrcnn_all_boxes')
    if self.provide_boxes_as_input:
      n_proposals = tf.shape(proposal_boxes)[0]
      pred_indices = tf.stack([tf.range(n_proposals), tf.zeros((n_proposals,), dtype=tf.int32)], axis=1)
      final_probs = tf.ones((n_proposals,))
      ## do not do a second bbox regression!
      #decoded_boxes = anchors
    else:
      # indices: Nx2. Each index into (#proposal, #category)
      pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs,
                                                       self._num_classes, self.result_score_thresh,
                                                       self.class_agnostic_box_and_mask_heads)
    final_probs = tf.identity(final_probs, 'final_probs')
    if self.class_agnostic_box_and_mask_heads:
      final_boxes = tf.gather(tf.squeeze(decoded_boxes, axis=1), pred_indices[:, 0], name='final_boxes')
    else:
      final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
    final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
    self.outputs_per_batch_idx[batch_idx] = [final_probs, final_boxes, final_labels]

    if self._add_maskrcnn:
      # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
      def f1():
        roi_resized = roi_align(rpn_input, final_boxes * (1.0 / ANCHOR_STRIDE), 14)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          feature_maskrcnn = add_resnet_conv5(roi_resized, self.tower_setup)[0]
        mask_logits = maskrcnn_head(
          feature_maskrcnn, self._num_classes, self.tower_setup,
          class_agnostic_conv=self.class_agnostic_box_and_mask_heads)  # #result x #cat x 14x14
        mask_logits = tf.transpose(mask_logits, [0, 3, 1, 2])
        if self.class_agnostic_box_and_mask_heads:
          mask_logits = tf.squeeze(mask_logits, axis=1)
        else:
          indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
          mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx14x14
        return tf.sigmoid(mask_logits)

      final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 14, 14]))
      final_masks = tf.identity(final_masks, name='final_masks')
      self.outputs_per_batch_idx[batch_idx].append(final_masks)
    else:
      final_masks = None

    if self._do_reid:
      final_reid_features = tf.gather(reid_features, pred_indices[:, 0])
      self.outputs_per_batch_idx[batch_idx].append(final_reid_features)
    else:
      final_reid_features = None

    return final_boxes, final_labels, final_masks, final_probs, final_reid_features

  def _add_basic_measures(self, inp, loss):
    n_examples = tf.shape(inp)[0]
    self.measures[Measures.N_EXAMPLES] = n_examples
    if loss is not None:
      self.measures[Measures.LOSS] = loss * tf.cast(n_examples, tf.float32)

  def _add_test_measures(self, batch_idx, final_boxes, final_probs, final_labels, final_masks, final_reid_features):
    if not self.is_training:
      orig_img_shape = self.network_input_dict[DataKeys.RAW_IMAGE_SIZES][batch_idx, ...]
      orig_img_shape_f = tf.cast(orig_img_shape, tf.float32)
      image_shape2d_f = tf.cast(self._image_shape2d, tf.float32)
      scale = (image_shape2d_f[0] / orig_img_shape_f[0] + image_shape2d_f[1] / orig_img_shape_f[1]) / 2
      boxes = final_boxes / scale
      clipped_boxes = clip_boxes(boxes, orig_img_shape_f)
      self.extractions_per_batch_idx[batch_idx][Extractions.DET_BOXES] = clipped_boxes
      self.extractions_per_batch_idx[batch_idx][Extractions.DET_PROBS] = final_probs
      self.extractions_per_batch_idx[batch_idx][Extractions.DET_LABELS] = final_labels
      self.extractions_per_batch_idx[batch_idx][Extractions.REID_FEATURES] = final_reid_features
      if self._add_maskrcnn:
        final_masks = tf.py_func(self.fill_full_mask, [clipped_boxes, final_masks, orig_img_shape],
                                 tf.uint8, name="fill_full_mask")
        self.extractions_per_batch_idx[batch_idx][Extractions.DET_MASKS] = final_masks
      if DataKeys.IMAGE_ID in self.network_input_dict:
        self.extractions_per_batch_idx[batch_idx][Extractions.IMAGE_ID] = self.network_input_dict[DataKeys.IMAGE_ID]


def create_reid_loss(reid_features_and_target_ids_per_time, reid_loss_per_class,
                     reid_loss_worst_examples_percent, loss_variant, loss_factor, distance_measure, adjacent_frames,
                     reid_loss_margin):
  if loss_variant == 0:
    assert False, "Sigmoid cross-entropy loss with adjacent frames was dropped"
  assert reid_loss_worst_examples_percent == 1.0, "Hard mining currently not implemented"

  if distance_measure == "cosine" or distance_measure == "normalized_euclidean":
    reid_features_and_target_ids_per_time = [(tf.nn.l2_normalize(reid, axis=1), ids)
                                             for (reid, ids) in reid_features_and_target_ids_per_time]

  def compute_measure(a, b):
    if distance_measure == "cosine":
      return 1 - tf.matmul(a, b, transpose_b=True)  # cosine similarity would be without 1-
    elif distance_measure == "euclidean" or distance_measure == "normalized_euclidean":
      return cdist(a, b, "euclidean")
    else:
      assert False, "Unknown measure for comparing reid vectors"

  id_fun = partial(_create_reid_loss_for_id, reid_loss_per_class=reid_loss_per_class,
                   reid_loss_worst_examples_percent=reid_loss_worst_examples_percent, loss_variant=loss_variant,
                   margin=reid_loss_margin)

  reid_loss = tf.constant(0.0)
  normalization = tf.constant(0)
  if adjacent_frames:
    # Here, we only look at adjacent pairs of frames, looking from t to t+1 (tp1)
    for (reid_features_t, target_ids_t), (reid_features_tp1, target_ids_tp1) in \
         zip(reid_features_and_target_ids_per_time, reid_features_and_target_ids_per_time[1:]):
      def f():
        computed_measure = compute_measure(reid_features_t, reid_features_tp1)
        # Finding hard pos/neg in this matrix is easier if we separate by id
        reid_loss_per_id_fn = partial(id_fun, computed_measure=computed_measure, target_ids_axis_0=target_ids_t,
                                      target_ids_axis_1=target_ids_tp1)
        unique_target_ids_t, _ = tf.unique(target_ids_t)
        reid_losses_per_id, normalization_per_id = tf.map_fn(reid_loss_per_id_fn, unique_target_ids_t,
                                                             dtype=(tf.float32, tf.int32))
        reid_loss_t = tf.reduce_sum(reid_losses_per_id, axis=0)
        normalization_t = tf.reduce_sum(normalization_per_id, axis=0)
        return reid_loss_t, normalization_t

      reid_loss_t, normalization_t = tf.cond(tf.logical_and(tf.size(reid_features_t) > 0,
                                                            tf.size(reid_features_tp1) > 0),
                                             f, lambda: (tf.constant(0.0), tf.constant(1, dtype=tf.int32)))
      reid_loss += reid_loss_t
      normalization += normalization_t
  else:
    all_target_ids = tf.concat([ids for (_, ids) in reid_features_and_target_ids_per_time], axis=0)
    all_reid_features = tf.concat([reid for (reid, _) in reid_features_and_target_ids_per_time], axis=0)

    computed_measure = compute_measure(all_reid_features, all_reid_features)
    # Finding hard pos/neg in this matrix is easier if we separate by id
    reid_loss_per_id_fn = partial(id_fun, computed_measure=computed_measure,
                                  target_ids_axis_0=all_target_ids, target_ids_axis_1=all_target_ids)
    unique_target_ids, _ = tf.unique(all_target_ids)
    reid_losses_per_id, normalization_per_id = tf.map_fn(reid_loss_per_id_fn, unique_target_ids,
                                                         dtype=(tf.float32, tf.int32))
    reid_loss = tf.reduce_sum(reid_losses_per_id, axis=0)
    normalization = tf.reduce_sum(normalization_per_id, axis=0)
    if loss_variant == 1:
      normalization = tf.size(all_target_ids)  # Sanity check: these should be equal
  reid_loss = reid_loss / tf.cast(normalization, dtype=tf.float32)
  return reid_loss * loss_factor


def _create_reid_loss_for_id(id_, computed_measure, target_ids_axis_0, target_ids_axis_1, reid_loss_per_class,
                             reid_loss_worst_examples_percent, loss_variant, margin):
  id_mask_axis_0 = tf.equal(target_ids_axis_0, id_)
  id_mask_axis_1 = tf.equal(target_ids_axis_1, id_)
  sliced_matrix = tf.boolean_mask(computed_measure, id_mask_axis_0)
  if reid_loss_per_class:
    # for the loss of each detection, consider only detections having the same class (car or pedestrian)
    class_mask_axis_1 = tf.equal(tf.floordiv(target_ids_axis_1, 1000), tf.floordiv(id_, 1000))
    sliced_matrix = tf.boolean_mask(sliced_matrix, class_mask_axis_1, axis=1)
    id_mask_axis_1 = tf.boolean_mask(id_mask_axis_1, class_mask_axis_1)
  same = tf.boolean_mask(sliced_matrix, id_mask_axis_1, axis=1)
  different = tf.boolean_mask(sliced_matrix, tf.logical_not(id_mask_axis_1), axis=1)

  def triplet():
    if loss_variant == 2 or loss_variant == 3:  # batch all
      # TODO in each row, leave out the column corresponding to the same detection
      all_combinations = tf.expand_dims(same, axis=2) - tf.expand_dims(different, axis=1)
      loss = tf.maximum(tf.constant(margin) + all_combinations, 0)
      if loss_variant == 2:
        normalization = tf.size(loss)
      else:  # batch all no zeros
        normalization = tf.count_nonzero(loss, dtype=tf.int32)
    else:  # batch hard
      hard_pos = tf.reduce_max(same, axis=1)
      hard_neg = tf.reduce_min(different, axis=1)
      loss = tf.maximum(tf.constant(margin) + hard_pos - hard_neg, 0)
      normalization = tf.size(loss)
    return tf.reduce_sum(loss), normalization

  def contrastive():
    loss = tf.constant(0.5) * (tf.reduce_sum(tf.square(same)) +
                               tf.reduce_sum(tf.square(tf.maximum(tf.constant(margin) - different, 0))))
    normalization = tf.size(same) + tf.size(different)
    return loss, normalization

  if loss_variant == 4:
    loss, normalization = tf.cond(tf.logical_and(tf.size(same) > 0, tf.size(different) > 0), contrastive,
                                  lambda: (tf.constant(0.0), tf.constant(1, dtype=tf.int32)))
  else:
    loss, normalization = tf.cond(tf.logical_and(tf.size(same) > 0, tf.size(different) > 0), triplet,
                                  lambda: (tf.constant(0.0), tf.constant(1, dtype=tf.int32)))
  return loss, normalization


# Stole this from https://raw.githubusercontent.com/VisualComputingInstitute/triplet-reid/master/loss.py :P
def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.
    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).
    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.
    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.
    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]