import time


class Timer:
  def __init__(self, message="", stream=None):
    if stream is None:
      from core.Log import log
      stream = log.v4
    self.stream = stream
    self.start_time = time.time()
    self.message = message

  def __enter__(self):
    self.start_time = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.message is not None:
      print(self.message, "elapsed", self.elapsed(), file=self.stream)

  def elapsed(self):
    end = time.time()
    start = self.start_time
    elapsed = end - start
    return elapsed
