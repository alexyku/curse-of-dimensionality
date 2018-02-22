import argparse
import os
import uuid

import numpy as np
from scipy.misc import imsave
from sklearn.externals import joblib
import theano

from dist import MultiDistribution
from lib.lazy_function import LazyFunction as lazy_function
import gan

parser = argparse.ArgumentParser(description='minified bigan generator')

parser.add_argument('--noise', default='u-200')
parser.add_argument('--noise_weight', type=str)
parser.add_argument('--noise_input_weight', action='store_true')
parser.add_argument('--k', type=float, default=1)
parser.add_argument('--net_fc', type=int, default=0)
parser.add_argument('--net_fc_dims', default='')
parser.add_argument('--nobn', action='store_true', help='No batch norm')
parser.add_argument('--bn_separate', action='store_true')
parser.add_argument('--nogain', action='store_true', help='No gain')
parser.add_argument('--log_gain', action='store_true', help='Learn log of gain')
parser.add_argument('--nolog_gain', action='store_false', dest="log_gain", default=False)
parser.add_argument('--nobias', action='store_true', help='No bias')
parser.add_argument('--no_decay_bias', action='store_true')
parser.add_argument('--no_decay_gain', action='store_true', default=False)
parser.add_argument('--gen_net')
parser.add_argument('--gen_net_size', type=int, default=64)
parser.add_argument('--deconv_nonlin', default='ReLU')
parser.add_argument('--deconv_ksize', type=int, default=5)
parser.add_argument('--feat_net_size', type=int, default=64)
parser.add_argument('--conv_nonlin', default='LReLU')
parser.add_argument('--net_fc_drop', type=float, default=0)
parser.add_argument('--cond_fc', type=int)
parser.add_argument('--cond_fc_dims')
parser.add_argument('--cond_fc_drop')
parser.add_argument('--dataset', default='imagenet')
parser.add_argument('--raw_size', type=int, default=72)
parser.add_argument('--crop_size', type=int, default=64)
parser.add_argument('--crop_resize', type=int)
parser.add_argument('--exp_dir', default='./exp/imagenet_1000_size72_u-200_bigan')
parser.add_argument('--resume', type=int, default=100)
parser.add_argument('--weights')

args = parser.parse_args(args=[])

args.net_fc_dims = [int(d) for d in args.net_fc_dims.split(',') if d]


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

bnkwargs = dict(bnkwargs=dict(
    batch_norm=(not args.nobn),
    gain=(not args.nogain) and (not args.nobn),
    log_gain=args.log_gain,
    bias=(not args.nobias),
))

crop = 64
args.crop_resize = crop


def color_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


def rescale(X, orig, new, in_place=False):
    assert len(orig) == len(new) == 2
    (a, b), (x, y) = ([float(b) for b in r] for r in (orig, new))
    assert b > a and y > x
    if (a, b) == (x, y):
        return X
    if not in_place:
        X = X.copy()
    if a != 0:
        X -= a
    scale = (y - x) / (b - a)
    if scale != 1:
        X *= scale
    if x != 0:
        X += x
    return X


def inverse_transform(X, crop=args.crop_resize):
    X = X.reshape(-1, 3, crop, crop).transpose(0, 2, 3, 1)
    return rescale(X, (-1, 1), (0, 1))


def gen_transform(gX):
    return rescale(gX, (0, 1), (-1, 1))  # dataset.native_range)

same = 1
grid_size = same, same  # one image

dist = MultiDistribution(
    np.prod(grid_size),
    args.noise,
    normalize=True,
    weights=args.noise_weight,
    weight_embed=args.noise_input_weight
)

Z = dist.placeholders

# generation
gen_kwargs = dict(args=args, dist=dist, nc=3, bnkwargs=bnkwargs,
                  gen_transform=gen_transform)
train_gen = gan.Generator(**gen_kwargs)
gXtest = gan.Generator(source=train_gen.net, mode='test', **gen_kwargs).data

_gen = lazy_function(Z, gXtest.value)

def load_params(model_dir, weight_prefix=None, resume_epoch=None,
                groups=dict(gen=train_gen.net.params())):
    weight_prefix = '%s/%d' % (model_dir, resume_epoch)
    for key, param_list in groups.iteritems():
        if len(param_list) == 0:
            continue
        path = '%s_%s_params.jl' % (weight_prefix, key)
        saved_params = joblib.load(path)
        for saved, shared in zip(saved_params, param_list):
            shared.set_value(saved)

# if __name__ == '__main__':
#     # load the weights
#     load_params(weight_prefix=args.weights, resume_epoch=args.resume)
#
#     # SAMPLE Z
#     np.random.seed(43) # for some reason I have to do this here rather than above
#     z = [floatX(np.random.uniform(-1,1,size=(np.prod(grid_size), 200)))]
#
#     # generate image
#     gX = _gen(*z)
#
#     # save image to file
#     filename = 'gen_samples.png'
#     color_grid_vis(inverse_transform(gX), grid_size, filename)
#     print 'SAVED SAMPLE!'


# this version returns an image as a numpy array
def _render_image(z=None, _gen=_gen):
    if z is None:
        z = [floatX(np.random.uniform(-1, 1, size=(np.prod(grid_size), 200)))]
    else:
        z = [floatX(z).reshape((1, 200))]

    # pass z through generator to render
    X = _gen(*z)
    # fix image up
    X = inverse_transform(X)
    h, w = X[0].shape[:2]
    img = np.zeros((64, 64, 3))
    for n, x in enumerate(X):
        j = n/1
        i = n%1
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    return img
# z = [floatX(np.random.uniform(-1,1,size=(1, 200)))]
# imsave('test_render_function.png',render_image(z))


# this version returns the path for the uuid file
def render_image(z=None, _gen=_gen):
    img = _render_image(z, _gen)
    # make sure there is a tmp directory
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # save image to tmp with uuid as filename
    filename = "{}.png".format(str(uuid.uuid4()))
    path = os.path.join('tmp', filename)
    imsave(path, img)
    # return the path to the image
    return path


if __name__ == '__main__':
    model_dir = '%s/models'%(args.exp_dir,)
    
    # load the weights
    load_params(model_dir, weight_prefix=args.weights, resume_epoch=args.resume)
    print("WEIGHTS LOADED...")

    z = [floatX(np.random.uniform(-1,1,size=(1, 200)))]
    print render_image(z)
