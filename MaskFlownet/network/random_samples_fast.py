import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import os
import time

class RandomSamplesFast(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        x = in_data[0]
        
        """
        s = mx.nd.arange(0,x.size,dtype="int32")
        s = mx.nd.random.shuffle(s)
        self.idx = s[0:in_data[1].size].reshape(
            in_data[1].shape[0],in_data[1].shape[1])
        """
        self.idx = mx.nd.random.randint(0,x.size,shape=(
            in_data[1].shape[0],in_data[1].shape[1]))
        self.assign(out_data[0], req[0], x[self.idx])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        dy = out_grad[0]
        x = in_data[0]
        dx = mx.nd.zeros(x.size)
        dx[self.idx.flatten(inplace=True)] = dy.flatten(inplace=True)
        self.assign(in_grad[0], req[0], dx)

@mx.operator.register("random_samples_fast")
class RandomSamplesFastProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(RandomSamplesFastProp, self).__init__()

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
        return RandomSamplesFast()
