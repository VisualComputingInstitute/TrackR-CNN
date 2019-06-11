from datasets.Loader import register_dataset
from datasets.Mapillary.MapillaryLike_instance import MapillaryLikeInstanceDataset
from datasets.util.Util import username

DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/mapillary/"
NAME = "mapillary_instance"


@register_dataset("mapillary_instance_full", resolution="full")
@register_dataset("mapillary_instance_half", resolution="half")
@register_dataset("mapillary_instance_quarter", resolution="quarter")
class MapillaryInstanceDataset(MapillaryLikeInstanceDataset):
  def __init__(self, config, subset, resolution):
    assert resolution in ("quarter", "half", "full"), resolution
    if resolution == "full":
      default_path = DEFAULT_PATH
    else:
      default_path = DEFAULT_PATH.replace("/mapillary/", "/mapillary_{}/".format(resolution))

    # there are 37 classes with instances in total

    # we excluded the following:
    #  8: construction--flat--crosswalk-plain -> doesn't really look like a useful object category
    # 34: object--bike-rack -> holes*
    # 45: object--support--pole -> very large and thin -> bounding box does not capture it well
    # 46: object--support--traffic-sign-frame -> holes*
    # 47: object--support--utility-pole -> holes*

    # further candidate for exclusion:
    #  0: animal--bird  -> usually very small

    # *: holes means that there are large "holes" in the object which usually are still annotated as part of the object
    # this will not work well together with laser, so we exclude them
    vehicle_ids = [52, 53, 54, 55, 56, 57, 59, 60, 61, 62]
    human_ids = [19, 20, 21, 22]
    animal_ids = [0, 1]
    object_ids = [32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 48, 49, 50, 51]
    crosswalk_zebra_id = [23]
    cat_ids_to_use = vehicle_ids + human_ids + animal_ids + object_ids + crosswalk_zebra_id

    super().__init__(config, subset, NAME, default_path, "datasets/Mapillary/", 256, cat_ids_to_use)
