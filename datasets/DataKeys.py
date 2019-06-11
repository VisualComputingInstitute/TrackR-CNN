# we cannot use an enum here as tensorflow needs ordered keys

SEGMENTATION_LABELS = "segmentation_labels"
SEGMENTATION_LABELS_ORIGINAL_SIZE = "segmentation_labels_original_size"
BBOX_GUIDANCE = "bbox_guidance"
UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE = "unsigned_distance_transform_guidance"
SIGNED_DISTANCE_TRANSFORM_GUIDANCE = "signed_distance_transform_guidance"
LASER_GUIDANCE = "laser_guidance"
IMAGES = "images"
RAW_IMAGES = "raw_images"
INPUTS = "inputs"
IMAGE_FILENAMES = "image_filenames"
BBOXES_y0x0y1x1 = "bboxes_y0x0y1x1"  # Order like in the tensorflow object detection API
BBOXES_x0y0x1y1 = "bboxes_x0y0x1y1"  # Alternative order, used for example in the frcnn implementation
BBOXES_TO_REFINE_x0y0x1y1 = "bboxes_to_refine_x0y0x1y1"
CLASSES = "classes"
IDS = "ids"  # bounding box number 1,2,3...  or 0 if no bounding box
CROP_BOXES_y0x0y1x1 = "crop_boxes_y0x0y1x1"
DT_METHOD = "dt_method"
RAW_SEGMENTATION_LABELS = "raw_segmentation_labels"
CLICK_MAPS = "clicks_maps"
NEG_CLICKS = "neg_clicks"
POS_CLICKS = "pos_clicks"
OBJ_TAGS = "obj_tags"
FEATUREMAP_LABELS = "featuremap_labels"
FEATUREMAP_BOXES = "featuremap_boxes"
IMAGE_ID = "image_id"
IS_CROWD = "is_crowd"
SEGMENTATION_MASK = "segmentation_mask"
SKIP_EXAMPLE = "skip_example"  # If true, filter the corresponding example from the datastream. Useful, when an example
                               # turns out to be bad during preprocessing.
RAW_IMAGE_SIZES = "raw_image_sizes"  # h, w
TIME_OFFSETS = "time_offsets"
SPACE_OFFSETS = "space_offsets"
SEGMENTATION_INSTANCE_LABELS = "segmentation_instance_labels"
N_VALID_IDS = "n_valid_ids"