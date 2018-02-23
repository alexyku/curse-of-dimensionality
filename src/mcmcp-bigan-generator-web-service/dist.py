from __future__ import division

import itertools
from operator import itemgetter

import numpy as np
import theano
import theano.tensor as T

from lib.rng import t_rng
from lib.theano_utils import floatX, sharedX

import net
from net import L, Output

def castFloatX(x):
  return T.cast(x, theano.config.floatX)

def concat(x):
  x = list(x)
  if len(x) == 0:
    return None
  if len(x) == 1:
    return x[0]
  return T.concatenate(x, axis=1)

class Distribution(object):
  placeholders = []

  def __init__(self):
    raise NotImplementedError()

  def sample(self, num=None):
    raise NotImplementedError()

  def recon_error(self, recon_logits):
    raise NotImplementedError()

  def logits_to_recon(self, recon_logits):
    return [Output(recon_logits, (self.num, self.recon_dim))]

  def logits_to_sample(self, recon_logits):
    return self.logits_to_recon(recon_logits)

  def kl_divergence(self, recon_logits):
    raise NotImplementedError()

  def l2distable(self, recon):
    return recon

  def norm_divisor(self):
    return self.sample_dim

class UniformDistribution(Distribution):
  nickname = 'u'
  default_limits = (-1, +1)

  def __init__(self, num, definition, limits=default_limits,
               internal_rng=False, name=None):
    assert len(limits) == 2
    assert limits[1] > limits[0]
    self.limits = tuple(float(l) for l in limits)
    self.span = limits[1] - limits[0]
    if len(definition) != 1:
      raise ValueError('definition should have 1 parameter (dim), not %d'
                       % len(definition))
    try:
      dim = int(definition[0])
    except ValueError:
      raise ValueError('non-integer dim: %s' % dim)
    self.recon_dim = self.sample_dim = dim
    self.num = num
    self.rangekw = dict(low=self.limits[0], high=self.limits[1])
    if internal_rng:
      self.placeholders = [t_rng.uniform(size=(num, dim), **self.rangekw)]
    else:
      self.placeholders = [T.matrix()]
    self.flat_data = [Output(self.placeholders[0], shape=(self.num, dim))]

  def sample(self, num=None):
    if num is None: num = self.num
    return [floatX(np.random.uniform(size=(num, self.sample_dim),
                                     **self.rangekw))]

  def recon_error(self, recon):
    sample = self.flat_data[0].value
    if self.limits[0] != 0:
      sample -= self.limits[0]
    if self.span != 1:
      sample /= self.span
    recon = T.nnet.sigmoid(recon)
    axes = range(1, recon.ndim)
    return T.nnet.binary_crossentropy(recon, sample).sum(axis=axes)

  def _scale_and_shift(self, recon, input_range):
    x, y = input_range
    a, b = self.limits
    if x != 0:
      recon -= x
    scale = (b - a) / (y - x)
    if scale != 1:
      recon *= scale
    if a != 0:
      recon += a
    return recon

  def logits_to_recon(self, recon_logits):
    shape = self.num, self.recon_dim
    recon = T.nnet.sigmoid(recon_logits)
    limits = 0, 1
    return [Output(self._scale_and_shift(recon, limits), shape)]

class UniformDistributionL2Error(UniformDistribution):
  nickname = 'ul'

  def recon_error(self, recon):
    sample = self.flat_data[0].value
    axes = range(1, recon.ndim)
    return 0.5 * ((recon - sample) ** 2).sum(axis=axes)

  def logits_to_recon(self, recon_logits):
    shape = self.num, self.recon_dim
    return [Output(recon_logits, shape)]

class UniformDistributionClampedL2Error(UniformDistributionL2Error):
  nickname = 'ulc'

  def logits_to_recon(self, recon_logits):
    def blend(mask, true, false):
      return mask * true + (1 - mask) * false
    shape = self.num, self.recon_dim
    # clamp recon in the dist. range
    recon = recon_logits
    recon = blend(recon < self.limits[0], self.limits[0], recon)
    recon = blend(recon > self.limits[1], self.limits[1], recon)
    return [Output(recon, shape)]

def chain(args):
  return list(itertools.chain(*args))

class MultiDistribution(Distribution):
  known_dists = frozenset(d for d in globals().values() if
                          type(d) == type and issubclass(d, Distribution) and
                          hasattr(d, 'nickname'))
  dist_nickname_to_type = {d.nickname: d for d in known_dists}
  assert len(known_dists) == len(dist_nickname_to_type), 'duplicate nicknames'

  def __init__(self, num, definition, uniform_error='scel',
               weights=None, normalize=True, weight_embed=True,
               internal_rng=False):
    self.internal_rng = internal_rng
    self.num = num
    self.dists = self._parse_def(definition)
    self.sample_dims = [d.sample_dim for d in self.dists]
    self.sample_dim = sum(self.sample_dims)
    self.recon_dims = [d.recon_dim for d in self.dists]
    self.recon_dim = sum(self.recon_dims)
    if not internal_rng:
      def cat_placeholders(a, b):
        bp = b.placeholders
        if not isinstance(bp, list):
          bp = [bp]
        return a + bp
      self.placeholders = reduce(cat_placeholders, self.dists, [])

    if weights is None:
      weights = [1] * len(self.dists)
    assert len(weights) == len(self.dists), \
        'weights length must be the # of dists'
    assert not any(w < 0 for w in weights), \
        'weights must have no negative entries'
    self.weights = weights
    if normalize:
      # divide each weight by corresponding norm_divisor; rescale to sum to 1
      sum_weights = sum(weights)
      assert sum_weights > 0, 'weights must have at least one nonzero entry'
      self.weights = [float(w) / sum_weights / d.norm_divisor()
                      for d, w in zip(self.dists, weights)]
    self.weight_embed = weight_embed

  def _parse_def(self, definition):
    dists = []
    for dist_def in definition.split('_'):
      dist_def = dist_def.strip()
      if not dist_def: continue
      params = dist_def.split('-')
      dist_nickname, dist_params = params[0], params[1:]
      try:
        dist_type = self.dist_nickname_to_type[dist_nickname]
      except KeyError:
        mapping = self.dist_nickname_to_type.items()
        mapping.sort(key=itemgetter(0))
        known_nicknames = "\n".join('\t{}\t{}'.format(nick, t.__name__)
                                    for nick, t in mapping)
        e = 'Unknown Distribution nickname "{}". Known Distributions:\n{}'
        print e.format(dist_nickname, known_nicknames)
        raise
      dist = dist_type(self.num, dist_params, internal_rng=self.internal_rng)
      dists.append(dist)
    return dists

  def sample(self, num=None):
    return chain([d.sample(num=num) for d in self.dists])

  def sample_feed_dict(self, sample=None):
    if sample is None:
      sample = self.sample()
    return dict(zip(self.placeholders, sample))

  def recon_slice(self, recon):
    recon_slices = []
    offset = 0
    for dim in self.recon_dims:
      next_offset = offset + dim
      recon_slices.append(recon[:, offset : next_offset])
      offset = next_offset
    return recon_slices

  def _apply_slices(self, agg, op_name, recon, skip_unimplemented=False):
    results = []
    for d, r in zip(self.dists, self.recon_slice(recon)):
      try:
        op = getattr(d, op_name)
      except NotImplementedError:
        if skip_unimplemented:
          print 'Warning: op "%s" not implemented for ' \
                'distribution: %s; skipping' % (op_name, d)
          continue
        else:
          raise
      results.append(op(r))
    return agg(results)

  def _cat_slices(self, *a, **k):
    def cat(slices):
      return concat(slices)
    return self._apply_slices(cat, *a, **k)

  def _list_slices(self, *a, **k):
    def identity(x): return x
    return self._apply_slices(identity, *a, **k)

  def _sum_slices(self, *a, **k):
    return self._apply_slices(sum, *a, **k)

  def _weighted_sum_slices(self, *a, **k):
    def weighted_sum(args):
      assert len(self.weights) == len(args)
      return sum([(a if w==1 else w*a) for a, w in zip(args, self.weights)])
    return self._apply_slices(weighted_sum, *a, **k)

  def recon_error(self, recon):
    return self._sum_slices('recon_error', recon)

  def weighted_recon_error(self, recon):
    return self._weighted_sum_slices('recon_error', recon)

  def l2distable(self, recon):
    return self._cat_slices('l2distable', recon)

  def kl_divergence(self, recon):
    return self._sum_slices('kl_divergence', recon)

  def logits_to_recon(self, recon_logits):
    return chain(self._list_slices('logits_to_recon', recon_logits))

  def logits_to_sample(self, recon_logits):
    return chain(self._list_slices('logits_to_sample', recon_logits))

  def norm_divisor(self):
    return sum(d.norm_divisor() for d in self.dists)

  def embed_data(self, include_dists=None):
    if include_dists is None:
      include_dists = [True] * len(self.dists)
    assert len(include_dists) == len(self.dists)
    weights = self.weights if self.weight_embed else ([1] * len(self.weights))
    dists = zip(include_dists, self.dists, weights)
    h = chain([d.flat_data for i, d, _ in dists if i])
    return h, self.dist_sized_slices(weights)

  def dist_sized_slices(self, input_list):
    assert len(input_list) == len(self.dists)
    return chain([[i] * len(d.flat_data)
                 for i, d in zip(input_list, self.dists)])
