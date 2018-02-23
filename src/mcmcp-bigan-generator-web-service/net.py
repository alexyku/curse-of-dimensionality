from __future__ import division

import sys
sys.path.append('..')

from collections import Counter, OrderedDict
from functools import partial
import itertools

import numpy as np
import theano
import theano.tensor as T

from lib.rng import py_rng, np_rng, t_rng
from lib.theano_utils import floatX, sharedX

from timeit import Timer

class Output(object):
    def __init__(self, value, shape=None, index_max=None):
        if isinstance(value, Output):
            raise TypeError("value may not be an Output")
        self.value = value
        if shape is None:
            try:
                shape = value.get_value().shape
            except AttributeError:
                try:
                    shape = value.shape
                    if isinstance(shape, theano.Variable):
                        shape = None
                except AttributeError:
                    pass
        if shape is not None:
            for s in list(shape) + ([] if (index_max is None) else [index_max]):
                assert isinstance(s, int)
                assert s >= 0
            shape = tuple(shape)
            assert len(shape) == value.ndim
        self.shape = shape
        if index_max is not None:
            assert isinstance(value, int) or str(value.dtype).startswith('int'), \
                ('if index_max is given, value must be integer-typed; '
                 'was: %s' % value.dtype)
            assert index_max == int(index_max)
            index_max = int(index_max)
            if index_max < 0:
                raise ValueError('index_max must be non-negative')
        self.index_max = index_max

    def __repr__(self):
        args = self.value, self.shape
        if self.index_max is not None:
            args += self.index_max,
            return 'Output(%s, shape=%s, index_max=%d)' % args
        return 'Output(%s, shape=%s)' % args

reparam = False
exp_reparam = False
def reparameterized_weights(w, g, epsilon=1e-8, nin_axis=None, exp=exp_reparam):
    for axis in nin_axis:
        assert isinstance(axis, int)
        assert 0 <= axis < w.ndim
    norm = T.sqrt(T.sqr(w).sum(axis=nin_axis, keepdims=True) + epsilon)
    if exp: g = T.exp(g)
    g_axes = list(reversed(xrange(g.ndim)))
    dimshuffle_pattern = ['x' if (axis in nin_axis) else g_axes.pop()
                          for axis in range(w.ndim)]
    assert not g_axes
    if 'x' in dimshuffle_pattern:
        g = g.dimshuffle(*dimshuffle_pattern)
    return g * w / norm

def castFloatX(x):
    return T.cast(x, theano.config.floatX)

def align_dims(a, b, axis):
    extra_dims = a.ndim - (axis + b.ndim)
    if extra_dims < 0:
        raise ValueError('Must have a.ndim >= axis + b.ndim.')
    if extra_dims > 0:
        order = ('x',) * axis + tuple(range(b.ndim)) + ('x',) * extra_dims
        b = b.dimshuffle(*order)
    return b

def bias_add(h, b, axis=1):
    return h + align_dims(h, b, axis)

def scale_mul(h, g, axis=1):
    return h * align_dims(h, g, axis)

def scale_div(h, g, axis=1):
    return h / align_dims(h, g, axis)

class Layer(object):
    def __init__(self, *inputs, **kwargs):
        self.net         = kwargs.pop('net', None)
        if self.net is None:
            self.net = L
        self.name        = kwargs.pop('name', None)
        if self.name is None:
            self.name = '(anonymous) ' + self.__class__.__name__
        self.weight_init = kwargs.pop('weight_init', 0.02)
        if isinstance(inputs, Output):
            # `inputs` may be a single `Output`, or an iterable of them.
            # canonicalize single `Output`s to single-element lists here.
            inputs = [inputs]
        for input in inputs:
            assert isinstance(input, Output)
            assert input.shape is not None
        outs = self.get_output(*inputs, **kwargs)
        if not isinstance(outs, tuple):
            outs = outs,
        outs = list(outs)
        for index, out in enumerate(outs):
            assert isinstance(out, Output)
            if out.shape is None:
                skip_types = theano.compile.sharedvalue.SharedVariable, np.ndarray
                input_dict = {i.value: np.zeros(i.shape, dtype=i.value.dtype)
                              for i in inputs
                              if not isinstance(i.value, skip_types)}
                out_shape = out.value.shape.eval(input_dict)
                outs[index] = Output(out.value, out_shape,
                                     index_max=out.index_max)
        self.output = tuple(outs)
        print '(%s) Creating outputs with shapes: %s' % \
            (self.name, ', '.join(str(o.shape) for o in self.output))
        if len(self.output) == 1:
            self.output = self.output[0]

    def get_output(self, *a, **k):
        """Layer subclasses should implement get_output."""
        raise NotImplementedError

    def add_param(self, value, prefix=None, **kwargs):
        name = '%s/%s' % (self.name, prefix)
        return self.net._add_param(name, value, layer_name=self.name, **kwargs)

    def weights(self, shape, stddev=None, reparameterize=reparam,
                nin_axis=None, exp_reparam=exp_reparam):
        if stddev is None:
            stddev = self.weight_init
        print 'weights: initializing weights with stddev = %f' % stddev
        if stddev == 0:
            value = np.zeros(shape)
        else:
            value = np_rng.normal(loc=0, scale=stddev, size=shape)
        w = self.add_param(value, prefix='w')
        if isinstance(nin_axis, int):
            nin_axis = [nin_axis]
        assert isinstance(nin_axis, list)
        if reparameterize:
            g_shape = [dim for axis, dim in enumerate(shape)
                       if axis not in nin_axis]
            f_init = np.zeros if exp_reparam else np.ones
            g = self.add_param(f_init(g_shape, dtype=theano.config.floatX),
                               prefix='w_scale')
            w = reparameterized_weights(w, g, exp=exp_reparam,
                                        nin_axis=nin_axis)
        return w
    def biases(self, dim):
        return self.add_param(np.zeros(dim), prefix='b')
    def gains(self, dim, init_value=1):
        return self.add_param(init_value * np.ones(dim), prefix='g')
    def bn_count(self):
        return self.add_param(np.zeros(()), prefix='count', learnable=False,
                              dtype='int')
    def bn_mean(self, dim):
        return self.add_param(np.zeros(dim), prefix='mean', learnable=False)
    def bn_var(self, dim):
        return self.add_param(np.zeros(dim), prefix='var', learnable=False)

class Identity(Layer):
    def get_output(self, *h):
        return h

class Reshape(Layer):
    def get_output(self, h, shape=None):
        assert shape is not None, 'shape is required'
        return Output(h.value.reshape(shape), index_max=h.index_max)

class EltwiseSum(Layer):
    def get_output(self, *H):
        assert len(H) > 0
        assert all(h.shape == H[0].shape for h in H)
        if len(H) == 1:
            return H[0]
        return Output(sum(h.value for h in H))

def conv_out_shape(in_shape, nout, ksize, stride, pad):
    assert len(in_shape) == 4
    out_size = ((s - ksize + 2 * pad) // stride + 1 for s in in_shape[2:])
    out_shape = (in_shape[0], nout) + tuple(out_size)
    return out_shape

def get_pad(pad, ksize):
    if pad == 'SAME':
        pad = (ksize - 1) // 2
    return pad

def conv_kwargs(stride, pad):
    assert isinstance(pad, int), 'pad must be an int'
    return dict(subsample=(stride, stride), border_mode=(pad, pad))

def deconv(h, w, subsample=(1, 1), border_mode=(0, 0), out_dims=None,
           conv_mode='conv'):
    if out_dims is None:
        out_dims = h.shape[2] * subsample[0], h.shape[3] * subsample[1]
    assert len(out_dims) == 2
    out_shape = (None, None) + out_dims
    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=out_shape, border_mode=border_mode, subsample=subsample)
    return op(w, h, out_dims)

class Deconv(Layer):
    def get_output(self, h, nout=None, ksize=1, stride=1, pad='SAME',
                   stddev=None):
        if nout is None:
            raise ValueError('nout must be provided')
        h, h_shape = h.value, h.shape
        out_shape_specified = isinstance(nout, tuple)
        pad = get_pad(pad, ksize)
        if out_shape_specified:
            out_shape = (h_shape[0],) + nout
            nout = nout[0]
        else:
            assert isinstance(nout, int)
            out_size = (stride * (s - 1) + ksize + (ksize % 2) - 2 * pad
                        for s in h_shape[2:])
            out_shape = (h_shape[0], nout) + tuple(out_size)
        nin = h_shape[1]
        if len(h_shape) == 2:
            h_shape += 1, 1
            h = h.reshape(*h_shape)
        expected_input_shape = conv_out_shape(in_shape=out_shape,
                nout=nin, ksize=ksize, stride=stride, pad=pad)
        if h_shape != expected_input_shape:
            raise ValueError(('deconv: input shape %s does not match expected '
                              'input shape %s for output shape %s')
                             % (h_shape, expected_input_shape, out_shape))
        W = self.weights((nin, nout, ksize, ksize), stddev=stddev,
                         nin_axis=[0, 2, 3])
        kwargs = conv_kwargs(stride, pad)
        kwargs.update(out_dims=out_shape[2:])
        out = deconv(h, W, **kwargs)
        return Output(out)

class FC(Layer):
    def get_output(self, h, nout=None, stddev=None,
                   reparameterize=reparam, exp_reparam=exp_reparam):
        h, h_shape, h_max = h.value, h.shape, h.index_max
        nin = np.prod(h_shape[1:], dtype=np.int) if (h_max is None) else h_max
        out_shape_specified = isinstance(nout, tuple)
        if out_shape_specified:
            out_shape = nout
        else:
            assert isinstance(nout, int)
            out_shape = nout,
        nout = np.prod(out_shape)
        nin_axis = [0]
        W = self.weights((nin, nout), stddev=stddev,
            reparameterize=reparameterize, nin_axis=nin_axis,
            exp_reparam=exp_reparam)
        if h_max is None:
            if h.ndim > 2:
                h = T.flatten(h, 2)
            out = T.dot(h, W)
        else:
            assert nin >= 1, 'FC: h.index_max must be >= 1; was: %s' % (nin,)
            assert h.ndim == 1
            out = W[h]
        return Output(out)

class FCMult(Layer):
    def get_output(self, h, W):
        h, h_shape, h_max = h.value, h.shape, h.index_max
        nin = np.prod(h_shape[1:], dtype=np.int) if (h_max is None) else h_max
        assert nin == W.shape[0]
        W = W.value
        if h_max is None:
            if h.ndim > 2:
                h = T.flatten(h, 2)
            out = T.dot(h, W)
        else:
            assert nin >= 1, 'FC: h.index_max must be >= 1; was: %s' % (nin,)
            assert h.ndim == 1
            out = W[h]
        return Output(out)

class Gain(Layer):
    def get_output(self, h, log_gain=False, axis=1):
        h, h_shape = h.value, h.shape
        init_value = 0 if log_gain else 1
        g = self.gains(h_shape[1], init_value=init_value)
        if log_gain:
            g = T.exp(g)
        out = scale_mul(h, g, axis=axis)
        return Output(out)

class Bias(Layer):
    def get_output(self, h, axis=1):
        b = self.biases(h.shape[axis])
        out = bias_add(h.value, b, axis=axis)
        return Output(out)

class BiasAdd(Layer):
    def get_output(self, h, b, axis=0):
        out = bias_add(h.value, b.value, axis=axis)
        return Output(out)

class BatchNorm(Layer):
    def get_output(self, h, u=None, s=None, use_ave=False, ave_frac=1,
                   epsilon=1e-8, log_var_move_ave=False,
                   var_bias_correction=True, ignore_moment_grads=False):
        no_grad = theano.gradient.disconnected_grad
        def move_ave_update(param, update, log_update=False):
            if log_update:
                new_param = ave_frac * param + T.log(update)
            else:
                new_param = ave_frac * param + update
            self.net.deploy_updates[param] = new_param
        h, h_shape = h.value, h.shape
        assert h.ndim >= 1
        axes = [0] + range(2, h.ndim)
        count = self.bn_count()
        if not use_ave:
            move_ave_update(count, 1)
        if u is None:
            mu = self.bn_mean(h_shape[1])
            if use_ave:
                u = castFloatX(mu / count)
            else:
                u = h.mean(axis=axes)
                move_ave_update(mu, u)
        if ignore_moment_grads:
            u = no_grad(u)
        h = bias_add(h, -u)
        if s is None:
            sigma = self.bn_var(h_shape[1])
            if use_ave:
                s = castFloatX(sigma / count)
                if log_var_move_ave:
                    s = T.exp(s)
            else:
                s = T.sqr(h).mean(axis=axes)
                if var_bias_correction:
                    n = h.shape[0] * T.prod(h.shape[2:])
                    nf = T.cast(n, theano.config.floatX)
                    # undo 1/n normalization; renorm by 1/(n-1) (unbiased var.)
                    s_unbiased = (nf / (nf - 1)) * s
                else:
                    s_unbiased = s
                move_ave_update(sigma, s_unbiased, log_update=log_var_move_ave)
        stdev = T.sqrt(s + epsilon)
        if ignore_moment_grads:
            stdev = no_grad(stdev)
        h = scale_div(h, stdev)
        return Output(h)

class Nonlinearity(Layer):
    def nonlin(self, op, h):
        out = op(h.value)
        return Output(out, h.shape)

class ReLU(Nonlinearity):
    def get_output(self, h):
        def relu(x):
            return (x + abs(x)) / 2
        return self.nonlin(relu, h)

class Sigmoid(Nonlinearity):
    def get_output(self, h):
        return self.nonlin(T.nnet.sigmoid, h)

class Scale(Nonlinearity):
    def get_output(self, h, scale=1):
        if scale == 1:
            return h
        return self.nonlin(lambda x: scale * x, h)

class L(object):
    layers = {k: v for k, v in globals().iteritems()
              if isinstance(v, type) and issubclass(v, Layer)}
    def __getattr__(self, attr):
        def layer_method(*args, **kwargs):
            return self.layers[attr](*args, **kwargs).output
        return layer_method
L = L()

def checked_update(target_map, source={}, **new_kwargs):
    for k, v in itertools.chain(source.iteritems(), new_kwargs.iteritems()):
        if k in target_map:
            raise ValueError('checked_update: key exists: %s' % k)
        target_map[k] = v

class Net(object):
    layer_types = {k: v for k, v in globals().iteritems()
                   if isinstance(v, type) and issubclass(v, Layer)}

    def __init__(self, source=None, name=None):
        self.name = name
        self.name_prefix = '' if (name is None) else ('%s/' % name)
        if source is not None:
            assert name == source.name

        self.loss = OrderedDict()

        self.is_agg_loss = OrderedDict()
        self.agg_loss_terms = OrderedDict()

        self.layers = OrderedDict()

        self.updates = OrderedDict()

        self.deploy_updates = OrderedDict()
        self.layer_count = Counter()
        self.reuse = source is not None
        self._params = OrderedDict()
        self.source_params = source._params if self.reuse else None

    def params(self):
        return [p for p, _ in self._params.itervalues()]

    def learnables(self):
        return [p for p, l in self._params.itervalues() if l]

    def learnable_keys(self):
        return [k for k, (_, l) in self._params.iteritems() if l]

    def add_deploy_updates(self, *args, **kwargs):
        for k in (dict(args), kwargs):
            checked_update(self.deploy_updates, k)

    def add_updates(self, *args, **kwargs):
        for k in (dict(args), kwargs):
            checked_update(self.updates, k)

    def get_updates(self, updater=None, loss='loss', extra_params=[]):
        updates = self.updates.items()
        if updater is not None:
            try:
                loss_value = self.get_loss(loss).mean()
                params = self.learnables() + extra_params
                updates += updater(params, loss_value)
            except KeyError:
                # didn't have a loss, check that we also had no learnables
                assert not self.learnables(), 'had no loss but some learnables'
        return updates

    def get_deploy_updates(self):
        return self.deploy_updates.items()

    def add_loss(self, value, weight=1, name='loss'):
        print 'Adding loss:', (self.name, weight, name)
        if value.ndim > 1:
            raise ValueError('value must be 0 or 1D (not %dD)' % value.ndim)
        if name not in self.is_agg_loss:
            self.is_agg_loss[name] = False
        assert not self.is_agg_loss[name]
        if (name not in self.loss) and (weight == 1):
            self.loss[name] = value
        else:
            if weight == 0:
                value = T.zeros_like(value, dtype=theano.config.floatX)
                self.loss[name] = value
            else:
                if weight != 1:
                    value *= weight
                if name in self.loss:
                    self.loss[name] += value
                else:
                    self.loss[name] = value

    def add_agg_loss_term(self, term_name, weight=1, name='loss'):
        print 'Adding agg loss:', (self.name, weight, name, term_name)
        if name not in self.is_agg_loss:
            self.is_agg_loss[name] = True
            self.agg_loss_terms[name] = []
        assert self.is_agg_loss[name]
        assert name != term_name
        self.agg_loss_terms[name].append((term_name, weight))

    def get_loss(self, name='loss'):
        if self.is_agg_loss[name]:
            return sum(w * self.get_loss(k).mean()
                       for k, w in self.agg_loss_terms[name])
        no_grad = theano.gradient.disconnected_grad
        total_loss = self.loss[name]
        assert total_loss.dtype.startswith('float')
        return total_loss

    def _add_layer(self, layer_constructor, *args, **kwargs):
        type_name = layer_constructor.__name__
        self.layer_count[type_name] += 1
        name = '%s%s%d' % (self.name_prefix, type_name,
                           self.layer_count[type_name])
        checked_update(kwargs, net=self, name=name)
        layer = layer_constructor(*args, **kwargs)
        checked_update(self.layers, **{name: layer})
        return layer

    def _add_param(self, name, value, learnable=True, layer_name='',
                   dtype=theano.config.floatX):
        if self.reuse:
            assert name in self.source_params, \
                'param "%s does not exist and self.reuse==True' % name
            param = self.source_params[name][0]
            existing_shape = param.get_value().shape
            if value.shape != existing_shape:
                raise ValueError('Param "%s": incompatible shapes %s vs. %s' %
                                 (name, existing_shape, value.shape))
            print '(%s) Reusing param "%s" with shape: %s' % \
                (layer_name, name, value.shape)
        else:
            print '(%s) Adding param "%s" with shape: %s' % \
                  (layer_name, name, value.shape)
            param = sharedX(value, dtype=dtype, name=name)
        assert name not in self._params, 'param "%s already exists' % name
        self._params[name] = (param, bool(learnable))
        return param

    def __getattr__(self, attr):
        def layer_method(*args, **kwargs):
            return self._add_layer(self.layer_types[attr],
                                   *args, **kwargs).output
        if attr in self.layer_types:
            return layer_method
        raise AttributeError('Unknown attribute: %s' % attr)

def batch_norm(N, h, batch_norm=True, bias=False, gain=False, log_gain=False,
               use_ave=False):
    if batch_norm: h = N.BatchNorm(h, use_ave=use_ave)
    if gain      : h = N.Gain(h, log_gain=log_gain)
    if bias      : h = N.Bias(h)
    return h

def multifc(N, H, nout=None, renormalize_weights=True, **kwargs):
    if isinstance(H, tuple):
        H, weights = H
    else:
        weights = None
    if isinstance(H, Output):
        H = [H]
        if weights is not None:
            weights = [weights]
    if weights is None:
        weights = [1] * len(H)
    assert isinstance(H, list) and isinstance(weights, list)
    assert len(H) == len(weights)
    for h in H:
        assert isinstance(h, Output)
    weights = np.array(weights, dtype=theano.config.floatX)
    if renormalize_weights:
        weights *= len(weights) / np.sum(weights)
    unweighted_outputs = [N.FC(h, nout=nout, **kwargs) for h in H]
    weighted_outputs = [N.Scale(o, scale=w)
                        for o, w in zip(unweighted_outputs, weights)]
    return N.EltwiseSum(*weighted_outputs)

def apply_cond(N, h, cond=None, ksize=1, bn=None, bn_separate=False):
    if cond is not None:
        stddev = 0.02
        if not bn_separate:
            stddev *= ksize ** 2
        b = multifc(N, cond, nout=h.shape[1], stddev=stddev)
        if (bn is not None) and bn_separate:
            b = bn(b)
            h = bn(h)
        h = N.BiasAdd(h, b)
        if (bn is not None) and bn_separate:
            scale = floatX(1. / np.sqrt(2))
            h = N.Scale(h, scale=scale)
    if (bn is not None) and ((not bn_separate) or (cond is None)):
        h = bn(h)
    return h

kwargs64 = dict(batch_norm=True, bias=True, gain=True)

def deconvnet_64(h, N=None, nout=3, size=None, bn_flat=True,
                 nonlin='ReLU', bnkwargs=kwargs64, num_fc=0, fc_dims=[],
                 bn_use_ave=False, num_refine=0, refine_ksize=5,
                 start_size=4, ksize=5, deconv_op='Deconv'):
    cond = h
    if N is None: N = Net()
    nonlin = getattr(N, nonlin)
    if size is None: size = 128
    def acts(h, ksize=1, do_cond=True):
        if do_cond: h = apply_cond(N, h, cond=cond, ksize=ksize)
        h = batch_norm(N, h, use_ave=bn_use_ave, **bnkwargs)
        h = nonlin(h)
        return h
    deconv_op = getattr(N, deconv_op)
    def deconv_acts(h, ksize=ksize, **kwargs):
        return acts(deconv_op(h, ksize=ksize, **kwargs), ksize=ksize)
    # do FCs
    fc_dims = [size*16] * num_fc + fc_dims
    for index, dim in enumerate(fc_dims):
        h = acts(multifc(N, h, nout=dim), do_cond=bool(index))
    # do deconv from 4x4
    ss = start_size
    shape = size*8, ss, ss
    if bn_flat:
        h = acts(multifc(N, h, nout=np.prod(shape)), do_cond=bool(fc_dims))
        channel_dim = np.prod(h.shape[1:]) // np.prod(shape[1:])
        assert channel_dim * np.prod(shape[1:]) == np.prod(h.shape[1:])
        shape = (channel_dim, ) + shape[1:]
        h = N.Reshape(h, shape=((-1, ) + shape))
    else:
        h = acts(multifc(N, h, nout=shape), do_cond=bool(fc_dims))
    h = deconv_acts(h, nout=(size*4, ss*2, ss*2), stride=2)
    h = deconv_acts(h, nout=(size*2, ss*4, ss*4), stride=2)
    h = deconv_acts(h, nout=(size*1, ss*8, ss*8), stride=2)
    curnout = (nout if num_refine == 0 else size//2, ss*16, ss*16)
    h = deconv_op(h, nout=curnout, ksize=ksize, stride=2)
    for i in xrange(num_refine):
        h = acts(h, ksize=k)
        is_last = (i == num_refine - 1)
        curnout = nout if is_last else (size//2)
        h = N.Conv(h, nout=curnout, ksize=refine_ksize, stride=1)
    h = N.Sigmoid(h) # generate images in [0, 1] range
    return h, N

def get_deconvnet(image_size=None, name=None):
    if name is None:
        assert image_size is not None
        name = 'deconvnet_%d' % image_size
    return globals()[name]
