import find_mxnet
import mxnet as mx
import logging
import data_processing as dp

def get_residual(data, nfilter):
    sym = mx.sym.Convolution(data, num_filter = nfilter,
                             kernel = (3, 3),
                             stride = (1, 1),
                             pad = (1, 1))
    sym = mx.sym.BatchNorm(sym, fix_gamma = False)
    sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    sym = mx.sym.Convolution(data, num_filter = nfilter,
                             kernel = (3, 3),
                             stride = (1, 1),
                             pad = (1, 1))
    sym = mx.sym.BatchNorm(sym, fix_gamma = False)
    return 0.5 * sym + 0.5 * data

# c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3
def get_generator(prefix, arch):
    blocks = arch.split(',')
    data = mx.sym.Variable('%s_data' % prefix)
    sym = data
    k = 0
    for block in blocks:
        k += 1
        logging.info('block: %s', block)
        if block.startswith('c'):
            kernel = int(block[1])
            stride = int(block[3])
            nfilter = int(block[5:])
            pad = kernel / 2
            sym = mx.sym.Convolution(sym, num_filter = nfilter,
                                     kernel = (kernel, kernel),
                                     stride = (stride, stride),
                                     pad = (pad, pad))
        elif block.startswith('d'):
            nfilter = int(block[1:])
            sym = mx.sym.Convolution(sym, num_filter = nfilter,
                                     kernel = (3, 3),
                                     stride = (2, 2),
                                     pad = (1, 1))
        elif block.startswith('u'):
            nfilter = int(block[1:])
            sym = mx.sym.Deconvolution(sym, num_filter = nfilter,
                                       kernel = (3, 3),
                                       stride = (2, 2),
                                       pad = (1, 1), 
                                       adj = (1, 1))
        elif block.startswith('R'):
            nfilter = int(block[1:])
            sym = get_residual(sym, nfilter)
        else:
            return None
        if k < len(blocks):
            sym = mx.sym.BatchNorm(sym, fix_gamma = False)
            sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    sym = mx.sym.Activation(sym, act_type = "tanh")
    sym *= 127
    return sym

def get_module(prefix, arch, dshape, ctx, is_train = True):
    sym = get_generator(prefix, arch)
    logging.info('sym ok')
    mod = mx.mod.Module(symbol = sym,
                        data_names = ('%s_data' % prefix,),
                        label_names = None,
                        context = ctx)
    logging.info('define module ok')
    if is_train:
        mod.bind(data_shapes = [('%s_data' % prefix, dshape)],
                 for_training = True,
                 inputs_need_grad = True)
    else:
        mod.bind(data_shapes = [('%s_data' % prefix, dshape)],
                 for_training = False,
                 inputs_need_grad = False)
    logging.info('module bind ok')
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    return mod
