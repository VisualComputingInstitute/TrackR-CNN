
EXTRACTIONS = "extractions"

CLASS_POSTERIORS = "class_posteriors"
SEGMENTATION_POSTERIORS = "segmentation_posteriors"
SEGMENTATION_MASK_ORIGINAL_SIZE = "segmentation_mask_original_size"
SEGMENTATION_MASK_INPUT_SIZE = "segmentation_mask_input_size"
RECURRENT_STATE = "recurrent_state"
REID_FEATURES = "reid_features"
DET_BOXES = "det_boxes"
DET_PROBS = "det_probs"
DET_LABELS = "det_labels"
DET_MASKS = "det_masks"
IMAGE_ID = "img_id"
CUSTOM_STRING = "custom_string"
BBOXES_y0x0y1x1_regressed = "bboxes_x0y0x1y1_regressed"
PRED_SPACE_TIME_OFFSETS = "pred_space_time_offsets"
PIXEL_POSITIONS = "pixel_positions"
FEATUREMAP_SIZE = "featuremap_size"


def accumulate_extractions(extractions_accumulator, *new_extractions):
  if len(new_extractions) == 0:
    return

  if len(extractions_accumulator) == 0:
    extractions_accumulator.update(new_extractions[0])
    new_extractions = new_extractions[1:]

  # each extraction will actually be a list, so we can just sum up the lists (extend the accumulator list with the next)
  for k, v in extractions_accumulator.items():
    for ext in new_extractions:
      extractions_accumulator[k] += ext[k]

  return extractions_accumulator


def extract_batch_size_1(extractions, key):
  if key not in extractions:
    return None
  val = extractions[key]

  # for now assume we only use 1 gpu for forwarding
  assert len(val) == 1, len(val)
  val = val[0]

  # for now assume, we use a batch size of 1 for forwarding
  assert val.shape[0] == 1, val.shape[0]
  val = val[0]

  return val
