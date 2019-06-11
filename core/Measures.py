import tensorflow as tf
import numpy as np

VOID_LABEL = 255

MEASURES = "measures"

N_EXAMPLES = "n_examples"
LOSS = "loss"
IOU = "IoU"
IOUs = "IoUs"
RECALL = "recall"
PRECISION = "precision"
MAP_BBOX = "mAP (bbox)"
MAP_BBOX_50 = "mAP (bbox) @50%"
MAP_SEGM = "mAP (segm)"
ERROR_RATE = "error_rate"
ERROR_RATE_AT_005 = "error_rate_at_005"
CONFUSION_MATRIX = "confusion_matrix"


def accumulate_measures(measures_accumulator, *new_measures, exclude=()):
  # this method will be called often so it should not be slow
  if len(new_measures) == 0:
    return

  if len(measures_accumulator) == 0:
    measures_accumulator.update(new_measures[0])
    new_measures = new_measures[1:]

  # most measures can just be summed up
  for k, v in measures_accumulator.items():
    if k not in exclude:
      for meas in new_measures:
        measures_accumulator[k] += meas[k]

  return measures_accumulator


def compute_measures_average(measures, for_final_result, exclude=()):
  measures_avg = {}
  n_examples = measures[N_EXAMPLES]
  # most measures can be handled by simple division by number of examples
  for k, v in measures.items():
    if k not in exclude:
      if k == CONFUSION_MATRIX:
        confmat = measures[k]
        n_classes = confmat.shape[1]
        ious = []
        for c in range(1, n_classes):
          i = confmat[c, c]
          u = confmat[c, :].sum() + confmat[:, c].sum() - i
          if u == 0:
            iou = float("nan")
          else:
            iou = i / u * 100
          ious.append(iou)
        measures_avg[IOUs] = np.array(ious)
      else:
        measures_avg[k] = measures[k] / n_examples
  del measures_avg[N_EXAMPLES]
  return measures_avg


def measures_string_to_print(measures, exclude=()):
  s = "{"
  first = True
  measures_to_print = [m for m in measures.keys() if m not in exclude]
  for k in sorted(measures_to_print):
    if k in (CONFUSION_MATRIX, IOUs):
      s += "{}{}: {}".format("" if first else ", ", k, measures[k])
    else:
      s += "{}{}: {:8.5}".format("" if first else ", ", k, measures[k])
    first = False
  s += "}"
  return s


def compute_measures_for_binary_segmentation_tf(predictions, targets):
  def f(ps, ts):
    meas = compute_measures_for_binary_segmentation_summed(ps, ts)
    # convert dict to list so it can be used by py_func
    meas = [np.cast[np.float32](meas[IOU]), np.cast[np.float32](meas[RECALL]), np.cast[np.float32](meas[PRECISION])]
    return meas
  res = tf.py_func(f, [predictions, targets], [tf.float32, tf.float32, tf.float32])
  for r in res:
    r.set_shape(())
  # convert into dict again
  res = {IOU: res[0], RECALL: res[1], PRECISION: res[2]}
  return res


def compute_measures_for_binary_segmentation_summed(predictions, targets):
  res = [compute_measures_for_binary_segmentation_single_image(p, t) for p, t in zip(predictions, targets)]
  accum = res[0]
  for r in res[1:]:
    for k, v in r.items():
      accum[k] += v
  return accum


def compute_measures_for_binary_segmentation_single_image(prediction, target):
  # we assume a single image here
  assert target.ndim == 2 or (target.ndim == 3 and target.shape[-1] == 1)
  valid_mask = target != VOID_LABEL
  T = np.logical_and(target, valid_mask).sum()
  P = np.logical_and(prediction, valid_mask).sum()
  I = np.logical_and(prediction == 1, target == 1).sum()
  U = np.logical_and(np.logical_or(prediction == 1, target == 1), valid_mask).sum()

  if U == 0:
    recall = 1.0
    precision = 1.0
    iou = 1.0
  else:
    if T == 0:
      recall = 1.0
    else:
      recall = float(I) / T

    if P == 0:
      precision = 1.0
    else:
      precision = float(I) / P

    iou = float(I) / U

  measures = {RECALL: recall, PRECISION: precision, IOU: iou}
  return measures
