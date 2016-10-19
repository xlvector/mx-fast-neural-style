import sys
import find_mxnet

import mxnet as mx
import numpy as np

#import basic
import data_processing
import gen, gen_v4
from skimage.filters import rank
from skimage.morphology import disk

dshape = (1, 3, 256, 256)
clip_norm = 1.0 * np.prod(dshape)
model_prefix = "./model/"
ctx = mx.gpu(0)

# generator
arch = 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3'
#arch = 'c5s1-48,c5s1-32,c3s1-64,c5s1-32,c5s1-48,c5s1-32,c3s1-3'
gen = gen.get_module('g', arch, dshape, ctx)
#gen = gen_v4.get_module("g", dshape, ctx)
gen.load_params(sys.argv[1])

content_np = data_processing.PreprocessContentImage(sys.argv[2], min(dshape[2:]), dshape)

data = [mx.nd.array(content_np)]
gen.forward(mx.io.DataBatch([data[-1]], [0]), is_train = False)
new_img = gen.get_outputs()[0]
data.append(new_img.copyto(mx.cpu()))
new_img = new_img.asnumpy()
data_processing.SaveImage(new_img, "out.jpg")

