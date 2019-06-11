import numpy as np

import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from scipy.misc import imsave
import sklearn.cluster
from sklearn.metrics import silhouette_score


def draw_box(ax, bbox, color, cat_id, alpha):
  import matplotlib.patches as patches
  rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                           edgecolor=color, facecolor='none', alpha=alpha)
  ax.add_patch(rect)
  ax.annotate(str(cat_id), (bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]), color=color, weight='bold',
              fontsize=10, ha='center', va='center', alpha=1.0)


# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] *
                              (1 - alpha) + alpha * color[c],
                              image[:, :, c])
  return image


class SilhouetteMiniBatchKMeans(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin, sklearn.base.TransformerMixin):

  def __init__(self, max_n_clusters=8, init='k-means++', max_iter=100,
               batch_size=100, verbose=0,
               random_state=None, tol=0.0, max_no_improvement=10,
               init_size=None, n_init=3, reassignment_ratio=0.01):
    self.clusterers = [sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters,
                                                      init=init, max_iter=max_iter, batch_size=batch_size,
                                                      verbose=verbose, compute_labels=True, random_state=random_state,
                                                      tol=tol, max_no_improvement=max_no_improvement,
                                                      init_size=init_size, n_init=n_init,
                                                      reassignment_ratio=reassignment_ratio)
                      for n_clusters in range(1, max_n_clusters+1)]
    self.max_n_clusters = max_n_clusters
    self.best_n_clusters = None

  def fit(self, X, y=None, sample_weight=None):
    silhouette_avgs = []
    for n_clusters in range(1, self.max_n_clusters):
      self.clusterers[n_clusters-1].fit(X, y, sample_weight)
      if n_clusters == 1:
        silhouette_avgs.append(-1.1)  # TODO
      else:
        silhouette_avgs.append(silhouette_score(X, self.clusterers[n_clusters-1].labels_))
    self.best_n_clusters = silhouette_avgs.index(max(silhouette_avgs)) + 1
    self.labels_ = self.clusterers[self.best_n_clusters-1].labels_
    self.cluster_centers_ = self.clusterers[self.best_n_clusters-1].cluster_centers_


'''
Adapted from https://github.com/nlml/np-to-tf-embeddings-visualiser/blob/master/save_embeddings.py
Function for easily going from np arrays to tensorboard embeddings visualiser.

Author: Liam Schoneveld
Also thanks to 
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
'''


def save_embeddings(images_features_labels, save_dir):
  '''
  Function to save embeddings (with corresponding labels and images) to a
      specified directory. Point tensorboard to that directory with
      tensorboard --logdir=<save_dir> and your embeddings will be viewable.
  Arguments:
  images_features_labels : dict
      each key in the dict should be the desired name for that embedding, and
      each element should be a list of [images, embeddings, labels] where
      images are a numpy array of images between 0. and 1. of shape [N*W*H*D]
      or [N*H*W] if grayscale (or None if no images), embeddings is a numpy
      array of shape [N*D], and labels is a numpy array of something that can
      be converted to string of shape D (or None if no labels available)
  save_dir : str
      path to save tensorboard checkpoints
  '''
  assert len(list(images_features_labels.keys())), 'Nothing in dictionary!'

  # Make directory if necessary
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # Reset graph and initialise file writer and session
  tf.reset_default_graph()
  writer = tf.summary.FileWriter(os.path.join(save_dir), graph=None)
  sess = tf.Session()
  config = projector.ProjectorConfig()

  # For each embedding name in the provided dictionary of embeddings
  for name in list(images_features_labels.keys()):

    [ims, fts, labs] = images_features_labels[name]

    # Save sprites and metadata
    if labs is not None:
      metadata_path = os.path.join(save_dir, name + '-metadata.tsv')
      save_metadata(labs, metadata_path)
    if ims is not None:
      sprites_path = os.path.join(save_dir, name + '.png')
      save_sprite_image(ims, path=sprites_path, invert=len(ims.shape) < 4)

    # Make a variable with the embeddings we want to visualise
    embedding_var = tf.Variable(fts, name=name, trainable=False)

    # Add this to our config with the image and metadata properties
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    if labs is not None:
      embedding.metadata_path = name + '-metadata.tsv'
    if ims is not None:
      embedding.sprite.image_path = name + '.png'
      embedding.sprite.single_image_dim.extend(ims[0].shape)

    # Save the embeddings
    projector.visualize_embeddings(writer, config)
  saver = tf.train.Saver(max_to_keep=1)
  sess.run(tf.global_variables_initializer())
  saver.save(sess, os.path.join(save_dir, 'ckpt'))


''' Functions below here inspired by / taken from:
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/'''


def create_sprite_image(images):
  """Returns a sprite image consisting of images passed as argument.
     Images should be count x width x height"""
  if isinstance(images, list):
    images = np.array(images)
  img_h = images.shape[1]
  img_w = images.shape[2]
  n_plots = int(np.ceil(np.sqrt(images.shape[0])))
  if len(images.shape) > 3:
    spriteimage = np.ones(
      (img_h * n_plots, img_w * n_plots, images.shape[3]))
  else:
    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
  four_dims = len(spriteimage.shape) == 4
  for i in range(n_plots):
    for j in range(n_plots):
      this_filter = i * n_plots + j
      if this_filter < images.shape[0]:
        this_img = images[this_filter]
        if four_dims:
          spriteimage[i * img_h:(i + 1) * img_h,
          j * img_w:(j + 1) * img_w, :] = this_img
        else:
          spriteimage[i * img_h:(i + 1) * img_h,
          j * img_w:(j + 1) * img_w] = this_img
  return spriteimage


def save_sprite_image(to_visualise, path, invert=True):
  if invert:
    to_visualise = invert_grayscale(to_visualise)
  sprite_image = create_sprite_image(to_visualise)
  imsave(path, sprite_image)  # , cmap='gray')


def invert_grayscale(mnist_digits):
  """ Makes black white, and white black """
  return 1 - mnist_digits


def save_metadata(batch_ys, metadata_path):
  with open(metadata_path, 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(batch_ys):
      if type(label) is int:
        f.write("%d\t%d\n" % (index, label))
      else:
        f.write('\t'.join((str(index), str(label))) + '\n')
