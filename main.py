#!/usr/bin/env python3
import sys
import os

from core.Engine import Engine
from core.Config import Config
from core.Log import log
import tensorflow as tf


def init_log(config):
  log_dir = config.dir("log_dir", "logs")
  model = config.string("model")
  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])


def main(_):
  assert len(sys.argv)==2 or len(sys.argv)==3, "usage: main.py <config> <optional_config_string>"
  config_path = sys.argv[1]
  assert os.path.exists(config_path), config_path
  try:
    if len(sys.argv)==3:
      config = Config(config_path, sys.argv[2])
    else:
      config = Config(config_path)
  except ValueError as e:
    print("Malformed config file:", e)
    return -1
  init_log(config)
  # dump the config into the log
  print(open(config_path).read(), file=log.v4)
  engine = Engine(config)
  engine.run()


if __name__ == '__main__':
  tf.app.run(main)
