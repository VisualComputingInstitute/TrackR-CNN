from abc import ABC, abstractmethod


class Forwarder(ABC):
  def __init__(self, engine):
    self.engine = engine
    self.config = engine.config
    self.session = engine.session
    self.val_data = self.engine.valid_data
    self.train_data = self.engine.train_data
    self.trainer = self.engine.trainer
    self.saver = self.engine.saver

  @abstractmethod
  def forward(self):
    pass
