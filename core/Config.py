import json
from collections import OrderedDict


class Config(object):
  def __init__(self, filename, update_config_string=""):
    lines = open(filename).readlines()
    # remove comments (lines starting with #)
    lines = [l if not l.strip().startswith("#") else "\n" for l in lines]
    s = "".join(lines)
    self._entries = json.loads(s, object_pairs_hook=OrderedDict)
    if update_config_string != "":
      config_string_entries = json.loads(update_config_string, object_pairs_hook=OrderedDict)
      print("Updating given config with dict", config_string_entries)
      self._entries.update(config_string_entries)

  def has(self, key):
    return key in self._entries

  def _value(self, key, dtype, default):
    if default is not None:
      assert isinstance(default, dtype)
    if key in self._entries:
      val = self._entries[key]
      if isinstance(val, dtype):
        return val
      else:
        raise TypeError()
    else:
      assert default is not None
      return default

  def _list_value(self, key, dtype, default):
    if default is not None:
      assert isinstance(default, list)
      for x in default:
        assert isinstance(x, dtype)
    if key in self._entries:
      val = self._entries[key]
      assert isinstance(val, list)
      for x in val:
        assert isinstance(x, dtype)
      return val
    else:
      assert default is not None
      return default

  def bool(self, key, default=None):
    return self._value(key, bool, default)

  def string(self, key, default=None):
    if isinstance(default, str):
      default = str(default)
    return self._value(key, str, default)

  def int(self, key, default=None):
    return self._value(key, int, default)

  def float(self, key, default=None):
    return self._value(key, float, default)

  def dict(self, key, default=None):
    return self._value(key, dict, default)

  def int_key_dict(self, key, default=None):
    if default is not None:
      assert isinstance(default, dict)
      for k in list(default.keys()):
        assert isinstance(k, int)
    dict_str = self.string(key, "")
    if dict_str == "":
      assert default is not None
      res = default
    else:
      res = eval(dict_str)
    assert isinstance(res, dict)
    for k in list(res.keys()):
      assert isinstance(k, int)
    return res

  def int_list(self, key, default=None):
    return self._list_value(key, int, default)

  def float_list(self, key, default=None):
    return self._list_value(key, float, default)

  def string_list(self, key, default=None):
    return self._list_value(key, str, default)

  def dir(self, key, default=None):
    p = self.string(key, default)
    if p[-1] != "/":
      return p + "/"
    else:
      return p
