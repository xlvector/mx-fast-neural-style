import find_mxnet
import mxnet as mx

#c32-9-1 : 32 * 9 * 9 conv with stride 1
def get_conv(data, nfilter, kernel, stride):
    sym = mx.sym.Convolution(data, num_filter = nfilter,
                             kernel = (kernel, kernel),
                             stride = stride,
                             pad = (kernel / 2, kernel / 2),
                             no_bias = False)
    sym = mx.sym.BatchNorm(sym, fix_gamma = False)
    sym = mx.sym.Activation(sym, act_type = "relu")
    return sym

def get_dconv(data, nfilter, kernel, stride):
    sym = mx.sym.Deconvolution(data, num_filter = nfilter,
                               kernel = (kernel, kernel),
                               stride = stride,
                               pad = (kernel / 2, kernel / 2),
                               no_bias = True)
    sym = mx.sym.BatchNorm(sym, fix_gamma = False)
    sym = mx.sym.Activation(sym, act_type = "relu")
    return sym

#r32-9
def get_residual(data, nfilter, kernel):
    sym = mx.sym.Convolution(data, num_filter = nfilter,
                             kernel = (kernel, kernel),
                             stride = 1,
                             pad = (kernel / 2, kernel / 2),
                             no_bias = False)
    sym = mx.sym.BatchNorm(sym, fix_gamma = False)
    sym = mx.sym.Activation(sym, act_type = "relu")
    sym = mx.sym.Convolution(data, num_filter = nfilter,
                             kernel = (kernel, kernel),
                             stride = 1,
                             pad = (kernel / 2, kernel / 2),
                             no_bias = False)
    sym = mx.sym.BatchNorm(sym, fix_gamma = False)
    return sym * 0.5 + data * 0.5

def get_generator(prefix, arch):
    blocks = arch.split(',')
    sym = mx.sym.Variable('%s_data' % prefix)
    for block in blocks:
        if block.startswith('c'):
            tks = block[1:].split('-')
            if len(tks) != 3:
                return None
            sym = get_conv(sym, int(tks[0]), int(tks[1]), int(tks[2]))
        elif block.startswith('d'):
            tks = block[1:].split('-')
            if len(tks) != 3:
                return None
            sym = get_dconv(sym, int(tks[0]), int(tks[1]), int(tks[2]))
        elif block.startswith('r'):
            tks = block[1:].split('-')
            if len(tks) != 2:
                return None
            sym = get_residual(sym, int(tks[0]), int(tks[1]))
        else:
            return None
    return sym

def get_module(prefix, arch, dshape, ctx, is_train = True):
    sym = get_generator(prefix, arch)
    mod = mx.mod.Module(symbol = sym,
                        data_names = ('%s_data' % prefix,),
                        label_names = None,
                        context = ctx)
    if is_train:
        mod.bind(data_shapes = [('%s_data' % prefix, dshape)],
                 for_training = True,
                 inputs_need_grad = True)
    else:
        mod.bind(data_shapes = [('%s_data' % prefix, dshape)],
                 for_training = False,
                 inputs_need_grad = False)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    return mod
