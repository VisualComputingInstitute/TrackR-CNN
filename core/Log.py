# this file is from RETURNN (https://github.com/rwth-i6/returnn/blob/master/Log.py)
import logging
import os
import sys

try:
  import StringIO
except ImportError:
  import io as StringIO
from threading import RLock


class Stream:
  def __init__(self, log, lvl):
    """
    :type log: logging.Logger
    :type lvl: int
    """
    self.buf = StringIO.StringIO()
    self.log = log
    self.lvl = lvl
    self.lock = RLock()

  def write(self, msg):
    with self.lock:
      if msg == '\n':
        self.flush()
      else:
        self.buf.write(msg)

  def flush(self):
    with self.lock:
      self.buf.flush()
      self.log.log(self.lvl, self.buf.getvalue())
      self.buf.truncate(0)
      # truncate does not change the current position.
      # In Python 2.7, it incorrectly does. See: http://bugs.python.org/issue30250
      self.buf.seek(0)


class Log(object):
  def initialize(self, logs=[], verbosity=[], formatter=[]):
    fmt = {'default': logging.Formatter('%(message)s'),
           'timed': logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%MS'),
           'raw''': logging.Formatter('%(message)s'),
           'verbose': logging.Formatter('%(levelname)s - %(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%MS')
           }
    self.v = [logging.getLogger('v' + str(v)) for v in range(6)]
    for l in self.v:
      # Reset handler list, in case we have initialized some earlier (e.g. multiple log.initialize() calls).
      l.handlers = []
    if not 'stdout' in logs:
      logs.append('stdout')
    for i in range(len(logs)):
      t = logs[i]
      v = 3
      if i < len(verbosity):
        v = verbosity[i]
      elif len(verbosity) == 1:
        v = verbosity[0]
      assert v <= 5, "invalid verbosity: " + str(v)
      f = fmt['default'] if i >= len(formatter) or not fmt.has_key(formatter[i]) else fmt[formatter[i]]
      if t == 'stdout':
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
      elif t.startswith("|"):  # pipe-format
        proc_cmd = t[1:].strip()
        from subprocess import Popen, PIPE
        proc = Popen(proc_cmd, shell=True, stdin=PIPE)
        handler = logging.StreamHandler(proc.stdin)
        handler.setLevel(logging.DEBUG)
      elif os.path.isdir(os.path.dirname(t)):
        handler = logging.FileHandler(t)
        handler.setLevel(logging.DEBUG)
      else:
        assert False, "invalid log target %r" % t
      handler.setFormatter(f)
      for j in range(v + 1):
        if not handler in self.v[j].handlers:
          self.v[j].addHandler(handler)
    self.verbose = [True] * 6
    null = logging.FileHandler(os.devnull)
    for i in range(len(self.v)):
      self.v[i].setLevel(logging.DEBUG)
      if not self.v[i].handlers:
        self.verbose[i] = False
        self.v[i].addHandler(null)
    self.error = Stream(self.v[0], logging.CRITICAL)
    self.v0 = Stream(self.v[0], logging.ERROR)
    self.v1 = Stream(self.v[1], logging.INFO)
    self.v2 = Stream(self.v[2], logging.INFO)
    self.v3 = Stream(self.v[3], logging.DEBUG)
    self.v4 = Stream(self.v[4], logging.DEBUG)
    self.v5 = Stream(self.v[5], logging.DEBUG)

  def write(self, msg):
    self.info(msg)


log = Log()

# Some external code (e.g. pyzmq) will initialize the logging system
# via logging.basicConfig(). That will set a handler to the root logger,
# if there is none. This default handler usually prints to stderr.
# Because all our custom loggers are childs of the root logger,
# this will have the effect that everything gets logged twice.
# By adding a dummy handler to the root logger, we will avoid that
# it adds any other default handlers.
logging.getLogger().addHandler(logging.NullHandler())
