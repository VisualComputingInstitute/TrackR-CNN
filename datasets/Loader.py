from core.Util import import_submodules

_registered_datasets = {}


def register_dataset(name, **args):
  name = name.lower()

  def _register(dataset):
    _registered_datasets[name] = (dataset, args)
    return dataset
  return _register


def load_dataset(config, subset, session, name):
  if not hasattr(load_dataset, "_imported"):
    load_dataset._imported = True
    import_submodules("datasets")
  name = name.lower()
  if name not in _registered_datasets:
    raise ValueError("dataset " + name + " not registered.")
  dataset, args = _registered_datasets[name]
  return dataset(config=config, subset=subset, **args)
