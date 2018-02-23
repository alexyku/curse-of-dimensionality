#!/usr/bin/env python

from net import Net, Output, get_deconvnet

class LearningModule(object):
    modes = frozenset(('train', 'test'))
    def set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError('Unknown mode %s; should be in: %s'
                             % (mode, self.modes))
        self.mode = mode

class Generator(LearningModule):
    def __init__(self, args, dist, nc, z=None, source=None, mode='train',
                 bnkwargs={}, gen_transform=None):
        N = self.net = Net(source=source, name='Generator')
        self.set_mode(mode)
        h_and_weights = dist.embed_data()
        bn_use_ave = (mode == 'test')
        self.data, _ = get_deconvnet(image_size=args.crop_resize,
                                     name=args.gen_net)(h_and_weights, N=N,
            nout=nc, size=args.gen_net_size, num_fc=args.net_fc,
            fc_dims=args.net_fc_dims, nonlin=args.deconv_nonlin,
            bn_use_ave=bn_use_ave, ksize=args.deconv_ksize, **bnkwargs)
        if gen_transform is not None:
            self.data = Output(gen_transform(self.data.value),
                               shape=self.data.shape)
