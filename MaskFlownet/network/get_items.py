import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import os

class GetItems(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        x = in_data[0]
        i = in_data[1].astype("int32")
        y = x[i]
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        dy = out_grad[0]
        x = in_data[0]
        i = in_data[1].astype("int32")
        dx = mx.nd.zeros(x.size)
        if len(dy.shape) < 2:
            dx[i] = dy
        else:
            raise NotImplementedError()
            print("Adding gradients...")
            for j in range(dy.shape[0]):
                dx[i[j,:]] = dx[i[j,:]] + dy[j,:]
        self.assign(in_grad[0], req[0], dx)
        #i.fill(0)
        #self.assign(in_grad[1], req[1], mx.nd.array(i))

@mx.operator.register("get_items")
class GetItemsProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GetItemsProp, self).__init__()

    def list_arguments(self):
        #  this can be omitted if you only have 1 input.
        return ['data','indices']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        data_shape = in_shapes[0]
        indices_shape = in_shapes[1]
        output_shape = indices_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape, indices_shape), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return GetItems()
