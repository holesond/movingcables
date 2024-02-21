import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import os
import time

class RandomIndicesInto(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        current_time = time.time()
        sub_sec = current_time - current_time//1
        sub_sec = sub_sec*1e5
        sub_sec = (int) (sub_sec)
        mx.random.seed(sub_sec)
        x = in_data[0]
        s = mx.nd.arange(0,x.size,dtype="int32")
        y = mx.nd.zeros((in_data[1].shape[0],in_data[1].shape[1]),
            dtype="int32")
        for i in range(in_data[1].shape[0]):
            s = mx.nd.random.shuffle(s)
            y[i,:] = mx.nd.slice(s,0,in_data[1].shape[1])
            assert(np.unique(y[i,:].asnumpy()).size == in_data[1].shape[1])
        #print(y)
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        pass
        #self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("random_indices_into")
class RandomIndicesIntoProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(RandomIndicesIntoProp, self).__init__()

    def list_arguments(self):
        #  this can be omitted if you only have 1 input.
        return ['data','shape_like']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        output_shape = in_shapes[1]
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (in_shapes[0],in_shapes[1]), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return RandomIndicesInto()
