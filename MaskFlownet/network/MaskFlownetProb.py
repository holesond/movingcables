from mxnet.gluon import nn
import math

from . import layer
from .MaskFlownet import Downsample, Upsample
from .MaskFlownet import use_bias
from . import flat_indices_like
from . import random_indices_into
from . import random_samples
from . import random_samples_fast
from . import get_items


class MaskFlownetSProb(nn.HybridBlock):
    def __init__(self, config = None, **kwargs):
        super().__init__(**kwargs)
        self.scale = 20. * config.network.flow_multiplier.get(1.)
        self.md = 4
        self.strides = [64, 32, 16, 8, 4]
        self.deform_bias = config.network.deform_bias.get(True)
        self.upfeat_ch = config.network.upfeat_ch.get([16, 16, 16, 16])

        with self.name_scope():
            #self.softplus = nn.Activation('softrelu')
            self.activate = nn.LeakyReLU(0.1)
            self.warp = layer.Reconstruction2D(2)

            self.conv1a = self.conv(  3,  16, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv1a')
            self.conv1b = self.conv( 16,  16, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv1b')
            self.conv1c = self.conv( 16,  16, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv1c')
            self.conv2a = self.conv( 16,  32, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv2a')
            self.conv2b = self.conv( 32,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2b')
            self.conv2c = self.conv( 32,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2c')
            self.conv3a = self.conv( 32,  64, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv3a')
            self.conv3b = self.conv( 64,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3b')
            self.conv3c = self.conv( 64,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3c')
            self.conv4a = self.conv( 64,  96, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv4a')
            self.conv4b = self.conv( 96,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4b')
            self.conv4c = self.conv( 96,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4c')
            self.conv5a = self.conv( 96, 128, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv5a')
            self.conv5b = self.conv(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5b')
            self.conv5c = self.conv(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5c')
            self.conv6a = self.conv(128, 196, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv6a')
            self.conv6b = self.conv(196, 196, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6b')
            self.conv6c = self.conv(196, 196, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6c')

            self.leakyRELU = nn.LeakyReLU(0.1)
            ch = None

            self.conv6_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_0')
            self.conv6_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_1')
            self.conv6_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_2')
            self.conv6_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_3')
            self.conv6_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_4')

            self.conv5_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_0')
            self.conv5_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_1')
            self.conv5_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_2')
            self.conv5_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_3')
            self.conv5_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_4')

            self.conv4_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_0')
            self.conv4_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_1')
            self.conv4_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_2')
            self.conv4_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_3')
            self.conv4_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_4')

            self.conv3_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_0')
            self.conv3_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_1')
            self.conv3_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_2')
            self.conv3_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_3')
            self.conv3_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_4')

            self.conv2_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_0')
            self.conv2_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_1')
            self.conv2_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_2')
            self.conv2_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_3')
            self.conv2_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_4')

            self.dc_conv1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1,  dilation=1,  prefix = 'dc_conv1')
            self.dc_conv2 = self.conv(ch, 128, kernel_size=3, stride=1, padding=2,  dilation=2,  prefix = 'dc_conv2')
            self.dc_conv3 = self.conv(ch, 128, kernel_size=3, stride=1, padding=4,  dilation=4,  prefix = 'dc_conv3')
            self.dc_conv4 = self.conv(ch,  96, kernel_size=3, stride=1, padding=8,  dilation=8,  prefix = 'dc_conv4')
            self.dc_conv5 = self.conv(ch,  64, kernel_size=3, stride=1, padding=16, dilation=16, prefix = 'dc_conv5')
            self.dc_conv6 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1,  dilation=1,  prefix = 'dc_conv6')
            self.dc_conv7 = self.predict_flow(ch, prefix = 'dc_conv7')

            self.upfeat5 = self.deconv(ch, self.upfeat_ch[0], kernel_size=4, stride=2, padding=1, prefix = 'upfeat5')
            self.upfeat4 = self.deconv(ch, self.upfeat_ch[1], kernel_size=4, stride=2, padding=1, prefix = 'upfeat4')
            self.upfeat3 = self.deconv(ch, self.upfeat_ch[2], kernel_size=4, stride=2, padding=1, prefix = 'upfeat3')
            self.upfeat2 = self.deconv(ch, self.upfeat_ch[3], kernel_size=4, stride=2, padding=1, prefix = 'upfeat2')

            self.pred_flow6 = self.predict_flow(ch, prefix = 'pred_flow6')
            self.pred_flow5 = self.predict_flow(ch, prefix = 'pred_flow5')
            self.pred_flow4 = self.predict_flow(ch, prefix = 'pred_flow4')
            self.pred_flow3 = self.predict_flow(ch, prefix = 'pred_flow3')
            self.pred_flow2 = self.predict_flow(ch, prefix = 'pred_flow2')

            self.pred_mask6 = self.predict_mask(ch, prefix = 'pred_mask6')
            self.pred_mask5 = self.predict_mask(ch, prefix = 'pred_mask5')
            self.pred_mask4 = self.predict_mask(ch, prefix = 'pred_mask4')
            self.pred_mask3 = self.predict_mask(ch, prefix = 'pred_mask3')

            self.deform5 = layer.DeformableConv2D(128, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform5')
            self.deform4 = layer.DeformableConv2D( 96, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform4')
            self.deform3 = layer.DeformableConv2D( 64, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform3')
            self.deform2 = layer.DeformableConv2D( 32, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform2')

            self.conv5f = self.conv(ch, 128, kernel_size = 3, stride = 1, activation = False, prefix = 'conv5f')
            self.conv4f = self.conv(ch,  96, kernel_size = 3, stride = 1, activation = False, prefix = 'conv4f')
            self.conv3f = self.conv(ch,  64, kernel_size = 3, stride = 1, activation = False, prefix = 'conv3f')
            self.conv2f = self.conv(ch,  32, kernel_size = 3, stride = 1, activation = False, prefix = 'conv2f')


    def conv(self, _, channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, activation = True, prefix = None):

        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(channels = channels, kernel_size = kernel_size, strides = stride, padding = padding, dilation = dilation, prefix = prefix))
            if activation:
                net.add(self.activate)

        return net

    def deconv(self, _, channels, kernel_size = 4, stride = 2, padding = 1, prefix = None):

        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2DTranspose(channels = channels, kernel_size = kernel_size, strides = stride, padding = padding, prefix = prefix))
            net.add(self.activate)

        return net

    def predict_flow(self, _, kernel_size = 3, stride = 1, padding = 1, prefix = None):

        return nn.Conv2D(4, kernel_size = kernel_size, strides = stride, padding = padding, prefix = prefix)

    def predict_mask(self, _, kernel_size = 3, stride = 1, padding = 1, prefix = None):

        return nn.Conv2D(1, kernel_size = kernel_size, strides = stride, padding = padding, prefix = prefix)

    def corr(self, F, im1, im2, stride1 = 1, stride2 = 1):

        return F.Correlation(im1, im2, pad_size = self.md, kernel_size = 1, max_displacement = self.md, stride1 = stride1, stride2 = stride2, is_multiply = 1)

    def hybrid_forward(self, F, im1, im2):
        c10 = im1
        c20 = im2

        c11 = self.conv1c(self.conv1b(self.conv1a(c10)))
        c21 = self.conv1c(self.conv1b(self.conv1a(c20)))
        c12 = self.conv2c(self.conv2b(self.conv2a(c11)))
        c22 = self.conv2c(self.conv2b(self.conv2a(c21)))
        c13 = self.conv3c(self.conv3b(self.conv3a(c12)))
        c23 = self.conv3c(self.conv3b(self.conv3a(c22)))
        c14 = self.conv4c(self.conv4b(self.conv4a(c13)))
        c24 = self.conv4c(self.conv4b(self.conv4a(c23)))
        c15 = self.conv5c(self.conv5b(self.conv5a(c14)))
        c25 = self.conv5c(self.conv5b(self.conv5a(c24)))
        c16 = self.conv6c(self.conv6b(self.conv6a(c15)))
        c26 = self.conv6c(self.conv6b(self.conv6a(c25)))


        warp6 = c26
        corr6 = self.corr(F, c16, warp6)
        corr6 = self.leakyRELU(corr6)
        x = corr6
        x = F.concat(self.conv6_0(x), x, dim=1)
        x = F.concat(self.conv6_1(x), x, dim=1)
        x = F.concat(self.conv6_2(x), x, dim=1)
        x = F.concat(self.conv6_3(x), x, dim=1)
        x = F.concat(self.conv6_4(x), x, dim=1)
        flow6ex = self.pred_flow6(x)
        flow6, flow6_var = F.split(flow6ex,axis=1,num_outputs=2)
        # flow6_var = self.softplus(flow6_var)
        mask6 = self.pred_mask6(x)

        feat5 = self.upfeat5(x)
        flow5 = Upsample(2)(flow6)
        flow5_var = Upsample(2)(flow6_var)
        flow5ex = F.concat(flow5, flow5_var, dim=1)
        mask5 = Upsample(2)(mask6)
        warp5 = self.deform5(c25, F.repeat(F.expand_dims(flow5*self.scale/self.strides[1], axis=1), 9, axis=1).reshape((0, -3, -2)))
        tradeoff5 = feat5
        warp5 = F.broadcast_mul(warp5, F.sigmoid(mask5)) + self.conv5f(tradeoff5)
        warp5 = self.leakyRELU(warp5)
        corr5 = self.corr(F, c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = F.concat(corr5, c15, feat5, flow5ex, dim=1)
        x = F.concat(self.conv5_0(x), x, dim=1)
        x = F.concat(self.conv5_1(x), x, dim=1)
        x = F.concat(self.conv5_2(x), x, dim=1)
        x = F.concat(self.conv5_3(x), x, dim=1)
        x = F.concat(self.conv5_4(x), x, dim=1)
        
        flow5ex = flow5ex + self.pred_flow5(x)
        flow5, flow5_var = F.split(flow5ex,axis=1,num_outputs=2)
        # flow5_var = self.softplus(flow5_var)
        mask5 = self.pred_mask5(x)

        feat4 = self.upfeat4(x)
        flow4 = Upsample(2)(flow5)
        flow4_var = Upsample(2)(flow5_var)
        flow4ex = F.concat(flow4, flow4_var, dim=1)
        mask4 = Upsample(2)(mask5)
        warp4 = self.deform4(c24, F.repeat(F.expand_dims(flow4*self.scale/self.strides[2], axis=1), 9, axis=1).reshape((0, -3, -2)))
        tradeoff4 = feat4
        warp4 = F.broadcast_mul(warp4, F.sigmoid(mask4)) + self.conv4f(tradeoff4)
        warp4 = self.leakyRELU(warp4)
        corr4 = self.corr(F, c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = F.concat(corr4, c14, feat4, flow4ex, dim=1)
        x = F.concat(self.conv4_0(x), x, dim=1)
        x = F.concat(self.conv4_1(x), x, dim=1)
        x = F.concat(self.conv4_2(x), x, dim=1)
        x = F.concat(self.conv4_3(x), x, dim=1)
        x = F.concat(self.conv4_4(x), x, dim=1)
        flow4ex = flow4ex + self.pred_flow4(x)
        flow4, flow4_var = F.split(flow4ex,axis=1,num_outputs=2)
        # flow4_var = self.softplus(flow4_var)
        mask4 = self.pred_mask4(x)

        feat3 = self.upfeat3(x)
        flow3 = Upsample(2)(flow4)
        flow3_var = Upsample(2)(flow4_var)
        flow3ex = F.concat(flow3, flow3_var, dim=1)
        mask3 = Upsample(2)(mask4)
        warp3 = self.deform3(c23, F.repeat(F.expand_dims(flow3*self.scale/self.strides[3], axis=1), 9, axis=1).reshape((0, -3, -2)))
        tradeoff3 = feat3
        warp3 = F.broadcast_mul(warp3, F.sigmoid(mask3)) + self.conv3f(tradeoff3)
        warp3 = self.leakyRELU(warp3)
        corr3 = self.corr(F, c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = F.concat(corr3, c13, feat3, flow3ex, dim=1)
        x = F.concat(self.conv3_0(x), x, dim=1)
        x = F.concat(self.conv3_1(x), x, dim=1)
        x = F.concat(self.conv3_2(x), x, dim=1)
        x = F.concat(self.conv3_3(x), x, dim=1)
        x = F.concat(self.conv3_4(x), x, dim=1)
        flow3ex = flow3ex + self.pred_flow3(x)
        flow3, flow3_var = F.split(flow3ex,axis=1,num_outputs=2)
        # flow3_var = self.softplus(flow3_var)
        mask3 = self.pred_mask3(x)

        feat2 = self.upfeat2(x)
        flow2 = Upsample(2)(flow3)
        flow2_var = Upsample(2)(flow3_var)
        flow2ex = F.concat(flow2, flow2_var, dim=1)
        mask2 = Upsample(2)(mask3)
        warp2 = self.deform2(c22, F.repeat(F.expand_dims(flow2*self.scale/self.strides[4], axis=1), 9, axis=1).reshape((0, -3, -2)))
        tradeoff2 = feat2
        warp2 = F.broadcast_mul(warp2, F.sigmoid(mask2)) + self.conv2f(tradeoff2)
        warp2 = self.leakyRELU(warp2)
        corr2 = self.corr(F, c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = F.concat(corr2, c12, feat2, flow2ex, dim=1)
        x = F.concat(self.conv2_0(x), x, dim=1)
        x = F.concat(self.conv2_1(x), x, dim=1)
        x = F.concat(self.conv2_2(x), x, dim=1)
        x = F.concat(self.conv2_3(x), x, dim=1)
        x = F.concat(self.conv2_4(x), x, dim=1)
        flow2ex = flow2ex + self.pred_flow2(x)
        
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2ex = flow2ex + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        flow2, flow2_var = F.split(flow2ex,axis=1,num_outputs=2)
        # flow2_var = self.softplus(flow2_var)
        
        predictions = [flow_ex * self.scale for flow_ex in [flow6ex,
            flow5ex, flow4ex, flow3ex, flow2ex]]
        occlusion_masks = []
        occlusion_masks.append(F.sigmoid(mask2))
        c1s = [c11, c12, c13, c14, c15, c16]
        c2s = [c21, c22, c23, c24, c25, c26]
        flows_ex = [flow6ex, flow5ex, flow4ex, flow3ex, flow2ex]
        mask0 = Upsample(4)(mask2)
        mask0 = F.sigmoid(mask0) - 0.5
        c30 = c10
        c40 = self.warp(c20, Upsample(4)(flow2)*self.scale)
        c30 = F.concat(c30, F.zeros_like(mask0), dim=1)
        c40 = F.concat(c40, mask0, dim=1)
        srcs = [c1s, c2s, flows_ex, c30, c40]
        return predictions, occlusion_masks, srcs


class MaskFlownetProb(nn.HybridBlock):
    def __init__(self, config = None, **kwargs):
        super().__init__(**kwargs)
        self.strides = [64, 32, 16, 8, 4]
        self.md = 2
        self.scale = 20. * config.network.flow_multiplier.get(1.)
        self.deform_bias = config.network.deform_bias.get(True)
        self.upfeat_ch = config.network.upfeat_ch.get([16, 16, 16, 16])

        with self.name_scope():
            # self.softplus = nn.Activation('softrelu')
            self.MaskFlownetSProb = MaskFlownetSProb(config)
            self.activate = nn.LeakyReLU(0.1)
            self.warp = layer.Reconstruction2D(2)

            self.conv1x = self.conv(  3,  16, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv1x')
            self.conv1y = self.conv( 16,  16, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv1y')
            self.conv1z = self.conv( 16,  16, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv1z')
            self.conv2x = self.conv( 16,  32, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv2x')
            self.conv2y = self.conv( 32,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2y')
            self.conv2z = self.conv( 32,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2z')
            self.conv3x = self.conv( 32,  64, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv3x')
            self.conv3y = self.conv( 64,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3y')
            self.conv3z = self.conv( 64,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3z')
            self.conv4x = self.conv( 64,  96, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv4x')
            self.conv4y = self.conv( 96,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4y')
            self.conv4z = self.conv( 96,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4z')
            self.conv5x = self.conv( 96, 128, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv5x')
            self.conv5y = self.conv(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5y')
            self.conv5z = self.conv(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5z')
            self.conv6x = self.conv(128, 196, kernel_size=3, stride=2, padding=1, dilation=1, prefix = 'conv6x')
            self.conv6y = self.conv(196, 196, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6y')
            self.conv6z = self.conv(196, 196, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6z')

            self.leakyRELU = nn.LeakyReLU(0.1)
            ch = None

            self.conv6_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_0')
            self.conv6_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_1')
            self.conv6_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_2')
            self.conv6_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_3')
            self.conv6_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv6_4')

            self.conv5_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_0')
            self.conv5_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_1')
            self.conv5_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_2')
            self.conv5_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_3')
            self.conv5_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv5_4')

            self.conv4_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_0')
            self.conv4_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_1')
            self.conv4_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_2')
            self.conv4_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_3')
            self.conv4_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv4_4')

            self.conv3_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_0')
            self.conv3_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_1')
            self.conv3_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_2')
            self.conv3_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_3')
            self.conv3_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv3_4')

            self.conv2_0 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_0')
            self.conv2_1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_1')
            self.conv2_2 = self.conv(ch,  96, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_2')
            self.conv2_3 = self.conv(ch,  64, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_3')
            self.conv2_4 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1, dilation=1, prefix = 'conv2_4')

            self.dc_conv1 = self.conv(ch, 128, kernel_size=3, stride=1, padding=1,  dilation=1,  prefix = 'dc_conv1')
            self.dc_conv2 = self.conv(ch, 128, kernel_size=3, stride=1, padding=2,  dilation=2,  prefix = 'dc_conv2')
            self.dc_conv3 = self.conv(ch, 128, kernel_size=3, stride=1, padding=4,  dilation=4,  prefix = 'dc_conv3')
            self.dc_conv4 = self.conv(ch,  96, kernel_size=3, stride=1, padding=8,  dilation=8,  prefix = 'dc_conv4')
            self.dc_conv5 = self.conv(ch,  64, kernel_size=3, stride=1, padding=16, dilation=16, prefix = 'dc_conv5')
            self.dc_conv6 = self.conv(ch,  32, kernel_size=3, stride=1, padding=1,  dilation=1,  prefix = 'dc_conv6')
            self.dc_conv7 = self.predict_flow(ch, prefix = 'dc_conv7')

            self.upfeat5 = self.deconv(ch, self.upfeat_ch[0], kernel_size=4, stride=2, padding=1, prefix = 'upfeat5')
            self.upfeat4 = self.deconv(ch, self.upfeat_ch[1], kernel_size=4, stride=2, padding=1, prefix = 'upfeat4')
            self.upfeat3 = self.deconv(ch, self.upfeat_ch[2], kernel_size=4, stride=2, padding=1, prefix = 'upfeat3')
            self.upfeat2 = self.deconv(ch, self.upfeat_ch[3], kernel_size=4, stride=2, padding=1, prefix = 'upfeat2')

            self.pred_flow6 = self.predict_flow(ch, prefix = 'pred_flow6')
            self.pred_flow5 = self.predict_flow(ch, prefix = 'pred_flow5')
            self.pred_flow4 = self.predict_flow(ch, prefix = 'pred_flow4')
            self.pred_flow3 = self.predict_flow(ch, prefix = 'pred_flow3')
            self.pred_flow2 = self.predict_flow(ch, prefix = 'pred_flow2')

            self.deform6 = layer.DeformableConv2D(196, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform6')
            self.deform5 = layer.DeformableConv2D(128, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform5')
            self.deform4 = layer.DeformableConv2D( 96, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform4')
            self.deform3 = layer.DeformableConv2D( 64, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform3')
            self.deform2 = layer.DeformableConv2D( 32, kernel_size=3, strides=1, padding=1, use_bias=self.deform_bias, prefix = 'deform2')


    def load_head(self, ckpt, ctx):
        self.MaskFlownetSProb.load_parameters(ckpt, ctx)

    def fix_head(self):
        for _, w in self.MaskFlownetSProb.collect_params().items():
            w.grad_req = 'null'

    def conv(self, _, channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, activation = True, prefix = None):
        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(channels = channels, kernel_size = kernel_size, strides = stride, padding = padding, dilation = dilation, prefix = prefix))
            if activation:
                net.add(self.activate)

        return net

    def deconv(self, _, channels, kernel_size = 4, stride = 2, padding = 1, prefix = None):
        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2DTranspose(channels = channels, kernel_size = kernel_size, strides = stride, padding = padding, prefix = prefix))
            net.add(self.activate)

        return net

    def predict_flow(self, _, kernel_size = 3, stride = 1, padding = 1, prefix = None):
        return nn.Conv2D(4, kernel_size = kernel_size, strides = stride, padding = padding, prefix = prefix)

    def predict_mask(self, _, kernel_size = 3, stride = 1, padding = 1, prefix = None):
        return nn.Conv2D(1, kernel_size = kernel_size, strides = stride, padding = padding, prefix = prefix)

    def corr(self, F, im1, im2, stride1 = 1, stride2 = 1):
        return F.Correlation(im1, im2, pad_size = self.md, kernel_size = 1, max_displacement = self.md, stride1 = stride1, stride2 = stride2, is_multiply = 1)

    def hybrid_forward(self, F, im1, im2):
        _, _, srcs = self.MaskFlownetSProb(im1, im2)
        c1s, c2s, flows_ex, c30, c40 = srcs
        c11, c12, c13, c14, c15, c16 = c1s
        c21, c22, c23, c24, c25, c26 = c2s

        c31 = self.conv1z(self.conv1y(self.conv1x(c30)))
        c32 = self.conv2z(self.conv2y(self.conv2x(c31)))
        c33 = self.conv3z(self.conv3y(self.conv3x(c32)))
        c34 = self.conv4z(self.conv4y(self.conv4x(c33)))
        c35 = self.conv5z(self.conv5y(self.conv5x(c34)))
        c36 = self.conv6z(self.conv6y(self.conv6x(c35)))

        c41 = self.conv1z(self.conv1y(self.conv1x(c40)))
        c42 = self.conv2z(self.conv2y(self.conv2x(c41)))
        c43 = self.conv3z(self.conv3y(self.conv3x(c42)))
        c44 = self.conv4z(self.conv4y(self.conv4x(c43)))
        c45 = self.conv5z(self.conv5y(self.conv5x(c44)))
        c46 = self.conv6z(self.conv6y(self.conv6x(c45)))


        flow6ex = flows_ex[0]
        flow6, flow6_var = F.split(flow6ex,axis=1,num_outputs=2)
        warp6u = self.deform6(c26, F.repeat(F.expand_dims(flow6*self.scale/self.strides[0], axis=1), 9, axis=1).reshape((0, -3, -2)))
        warp6u = self.leakyRELU(warp6u)
        corr6u = self.leakyRELU(self.corr(F, c16, warp6u))
        warp6v = c46
        corr6v = self.leakyRELU(self.corr(F, c36, warp6v))
        x = F.concat(corr6u, corr6v, flow6ex, dim=1)
        x = F.concat(self.conv6_0(x), x, dim=1)
        x = F.concat(self.conv6_1(x), x, dim=1)
        x = F.concat(self.conv6_2(x), x, dim=1)
        x = F.concat(self.conv6_3(x), x, dim=1)
        x = F.concat(self.conv6_4(x), x, dim=1)
        flow6ex = flow6ex + self.pred_flow6(x)
        flow6, flow6_var = F.split(flow6ex,axis=1,num_outputs=2)
        # flow6_var = self.softplus(flow6_var)

        feat5 = self.upfeat5(x)
        flow5 = Upsample(2)(flow6)
        flow5_var = Upsample(2)(flow6_var)
        flow5ex = F.concat(flow5, flow5_var, dim=1)
        warp5u = self.deform5(c25, F.repeat(F.expand_dims(flow5*self.scale/self.strides[1], axis=1), 9, axis=1).reshape((0, -3, -2)))
        warp5u = self.leakyRELU(warp5u)
        corr5u = self.leakyRELU(self.corr(F, c15, warp5u))
        warp5v = c45
        corr5v = self.leakyRELU(self.corr(F, c35, warp5v))
        x = F.concat(c15, feat5, corr5u, corr5v, flow5ex, flows_ex[1], dim=1)
        x = F.concat(self.conv5_0(x), x, dim=1)
        x = F.concat(self.conv5_1(x), x, dim=1)
        x = F.concat(self.conv5_2(x), x, dim=1)
        x = F.concat(self.conv5_3(x), x, dim=1)
        x = F.concat(self.conv5_4(x), x, dim=1)
        flow5ex = flow5ex + self.pred_flow5(x)
        flow5, flow5_var = F.split(flow5ex,axis=1,num_outputs=2)
        # flow5_var = self.softplus(flow5_var)

        feat4 = self.upfeat4(x)
        flow4 = Upsample(2)(flow5)
        flow4_var = Upsample(2)(flow5_var)
        flow4ex = F.concat(flow4, flow4_var, dim=1)
        warp4u = self.deform4(c24, F.repeat(F.expand_dims(flow4*self.scale/self.strides[2], axis=1), 9, axis=1).reshape((0, -3, -2)))
        warp4u = self.leakyRELU(warp4u)
        corr4u = self.leakyRELU(self.corr(F, c14, warp4u))
        warp4v = c44
        corr4v = self.leakyRELU(self.corr(F, c34, warp4v))
        x = F.concat(c14, feat4, corr4u, corr4v, flow4ex, flows_ex[2], dim=1)
        x = F.concat(self.conv4_0(x), x, dim=1)
        x = F.concat(self.conv4_1(x), x, dim=1)
        x = F.concat(self.conv4_2(x), x, dim=1)
        x = F.concat(self.conv4_3(x), x, dim=1)
        x = F.concat(self.conv4_4(x), x, dim=1)
        flow4ex = flow4ex + self.pred_flow4(x)
        flow4, flow4_var = F.split(flow4ex,axis=1,num_outputs=2)
        # flow4_var = self.softplus(flow4_var)

        feat3 = self.upfeat3(x)
        flow3 = Upsample(2)(flow4)
        flow3_var = Upsample(2)(flow4_var)
        flow3ex = F.concat(flow3, flow3_var, dim=1)
        warp3u = self.deform3(c23, F.repeat(F.expand_dims(flow3*self.scale/self.strides[3], axis=1), 9, axis=1).reshape((0, -3, -2)))
        warp3u = self.leakyRELU(warp3u)
        corr3u = self.leakyRELU(self.corr(F, c13, warp3u))
        warp3v = c43
        corr3v = self.leakyRELU(self.corr(F, c33, warp3v))
        x = F.concat(c13, feat3, corr3u, corr3v, flow3ex, flows_ex[3], dim=1)
        x = F.concat(self.conv3_0(x), x, dim=1)
        x = F.concat(self.conv3_1(x), x, dim=1)
        x = F.concat(self.conv3_2(x), x, dim=1)
        x = F.concat(self.conv3_3(x), x, dim=1)
        x = F.concat(self.conv3_4(x), x, dim=1)
        flow3ex = flow3ex + self.pred_flow3(x)
        flow3, flow3_var = F.split(flow3ex,axis=1,num_outputs=2)
        # flow3_var = self.softplus(flow3_var)

        feat2 = self.upfeat2(x)
        flow2 = Upsample(2)(flow3)
        flow2_var = Upsample(2)(flow3_var)
        flow2ex = F.concat(flow2, flow2_var, dim=1)
        warp2u = self.deform2(c22, F.repeat(F.expand_dims(flow2*self.scale/self.strides[4], axis=1), 9, axis=1).reshape((0, -3, -2)))
        warp2u = self.leakyRELU(warp2u)
        corr2u = self.leakyRELU(self.corr(F, c12, warp2u))
        warp2v = c42
        corr2v = self.leakyRELU(self.corr(F, c32, warp2v))
        x = F.concat(c12, feat2, corr2u, corr2v, flow2ex, flows_ex[4], dim=1)
        x = F.concat(self.conv2_0(x), x, dim=1)
        x = F.concat(self.conv2_1(x), x, dim=1)
        x = F.concat(self.conv2_2(x), x, dim=1)
        x = F.concat(self.conv2_3(x), x, dim=1)
        x = F.concat(self.conv2_4(x), x, dim=1)
        flow2ex = flow2ex + self.pred_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2ex = flow2ex + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        flow2, flow2_var = F.split(flow2ex,axis=1,num_outputs=2)
        # flow2_var = self.softplus(flow2_var)

        preds = [flow_ex * self.scale for flow_ex in [flow6ex,
            flow5ex, flow4ex, flow3ex, flow2ex]]
        visuals = []
        visuals.append(flow2.slice_axis(axis = 1, begin = 0, end = 1))
        return preds, visuals, []


class EpeLossProb(nn.HybridBlock):
    ''' Compute Endpoint Error Loss
    Arguments
    ==============
    - pred [N, C, H, W] : predictions
    - label [N, C, H, W] : flow_groundtruth
    '''
    def __init__(self, eps = 0, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def hybrid_forward(self, F, pred, label):
        flow, flow_var = F.split(pred,axis=1,num_outputs=2)
        loss = F.sqrt(F.sum(F.square(flow - label), axis=1) + self.eps)
        return F.mean(loss, axis=0, exclude=True)
        
class WassersteinCalLossProb(nn.HybridBlock):
    ''' Compute Wassestein Calibration Loss
    Arguments
    ==============
    - pred [N, C, H, W] : predictions
    - label [N, C, H, W] : flow_groundtruth
    '''
    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        
    def faster_sampling(self, F, residuals, dof, cs_samples):
        picked = F.zeros((cs_samples,dof))
        if False:
            index_sample = F.Custom(residuals, picked, op_type='random_indices_into')
            #indices = F.Custom(residuals, op_type='flat_indices_like')
            #index_sample = F.slice(F.random.shuffle(indices), 0, dof*cs_samples)
            index_sample = F.stop_gradient(index_sample)
            picked = F.Custom(residuals, index_sample, op_type='get_items')
        picked = F.Custom(residuals, F.stop_gradient(picked),
            op_type='random_samples_fast') # 'random_samples'
        #picked = F.reshape(picked, (dof, cs_samples))
        #print(index_sample)
        chi_sq_samples = F.sum(picked**2, axis=1)
        return chi_sq_samples
        
    def slower_sampling(self, F, residuals, dof, cs_samples):
        indices = F.Custom(residuals, op_type='flat_indices_like')
        index_sample = F.slice(F.random.shuffle(indices), 0, dof)
        index_sample = F.stop_gradient(index_sample)
        picked = F.Custom(residuals, index_sample, op_type='get_items')
        sample0 = F.sum(picked**2)
        chi_sq_samples = sample0
        
        for i in range(cs_samples-1):
            index_sample = F.slice(F.random.shuffle(indices), 0, dof)
            index_sample = F.stop_gradient(index_sample)
            picked = F.Custom(residuals, index_sample, op_type='get_items')
            sample = F.sum(picked**2)
            chi_sq_samples = F.concat(chi_sq_samples, sample, dim=0)
        if False:
            indices = F.Custom(residuals, op_type='flat_indices_like')
            index_sample = F.slice(F.random.shuffle(indices), 0, dof)
            picked = residuals[index_sample]
            sample0 = F.sum(picked**2)
            chi_sq_samples = sample0
            for i in range(cs_samples-1):
                index_sample = F.slice(F.random.shuffle(indices), 0, dof)
                sample = F.sum(residuals[index_sample]**2)
                chi_sq_samples = F.concat(chi_sq_samples, sample, dim=0)
        return chi_sq_samples

    def hybrid_forward(self, F, pred, label):
        flow, flow_var = F.split(pred,axis=1,num_outputs=2)
        flow, flow_var = F.reshape(flow,(-1,)), F.reshape(flow_var,(-1,))
        label = F.reshape(label,(-1,))
        ## CDF of laplace distribution
        uni_var = (label <= flow) * 0.5 * F.exp( F.minimum(label - flow,0) / F.maximum(flow_var,0.5) ) + \
            (label > flow) * 0.5 * (1 - 0.5 * F.exp(F.minimum(-(label - flow),0) / F.maximum(flow_var,0.5)))
        ## this is for numerical stability
        uniform_samples = F.clip(uni_var, 0.0000002, 0.9999998)
        SQRT2 = 1.41421356
        ## inverse CDF of standard normal distribution
        residuals = 0.0 + 1.0 * F.erfinv(2 * uniform_samples - 1) * SQRT2
        
        dof = 75    ## 75 degrees of freedom
        cs_samples = 100    ## 100 number of chi-squared samples
        chi_sq_samples = self.faster_sampling(F, residuals, dof, cs_samples)
        mu2, var2 = F.moments(chi_sq_samples,axes=[0])
        mu1, var1 = dof, 2*dof
        
        wasserstein_loss = ((mu1 - mu2)**2 + var1 + var2 - 2*(var1*var2+self.eps)**0.5)
        return wasserstein_loss
        
class NllLossWithMaskProb(nn.HybridBlock):
    ''' Compute Negative log likelihood Endpoint Error Loss
    Arguments
    ==============
    - pred [N, C, H, W] : predictions
    - label [N, C, H, W] : flow_groundtruth
    - mask [N, 1, H, W] : mask_groundtruth
    '''
    def __init__(
            self, eps=1e-8, q=None, min_var=1e-2, max_var=1e2,
            softplus_var=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.q = q
        self.min_var = min_var    # 0.5
        self.max_var = max_var
        self.min_var_exp_log = math.log(math.exp(self.min_var)-1.0)
        self.softplus_var = softplus_var
        with self.name_scope():
            self.softplus = nn.Activation('softrelu')

    def hybrid_forward(self, F, pred, label, mask):
        flow, flow_var_net = F.split(pred, axis=1, num_outputs=2)
        if self.softplus_var:
            flow_var = self.softplus(flow_var_net) + self.min_var
            # tmp = F.minimum(flow_var_net, 10.0)
            # tmp = F.maximum(tmp, -10.0)
            # flow_var = F.exp(tmp) + self.min_var
        else:
            if self.q is None:
                flow_var = F.maximum(flow_var_net, self.min_var**2)
            else:
                flow_var = F.maximum(flow_var_net, self.min_var)
        if self.q is not None:
            loss = (F.sum(F.abs(flow - label)/flow_var, axis = 1) + self.eps) ** self.q
            loss = loss + F.sum(F.log(2*flow_var), axis=1)
            # Laplace error distribution.
        else:
            loss = F.sqrt(
                F.sum(F.square(flow - label)/F.square(flow_var), axis = 1)
                + self.eps)
            loss = loss + F.sum(F.log(F.square(flow_var)), axis=1)
            # Multivariate Laplace error distribution.
        if not self.softplus_var:
            # Force flow_var_net to be non-negative, at least ca. self.min_var.
            loss = loss + F.sum(
                20*F.maximum(-flow_var_net+self.min_var, 0.0), axis=1)
        else:
            pass
            #loss = loss + F.sum(
            #    100*F.maximum(-flow_var_net+self.min_var_exp_log, 0.0), axis=1)
            #loss = loss + F.sum(
            #    100*F.maximum(flow_var_net-self.max_var, 0.0), axis=1)
        loss = F.broadcast_mul(loss, mask.squeeze(axis = 1))
        loss = F.sum(loss, axis=0, exclude=True) / F.sum(mask, axis=0, exclude=True)
        return loss


class EpeLossWithMaskProb(nn.HybridBlock):
    ''' Compute Endpoint Error Loss
    Arguments
    ==============
    - pred [N, C, H, W] : predictions
    - label [N, C, H, W] : flow_groundtruth
    - mask [N, 1, H, W] : mask_groundtruth
    '''
    def __init__(self, eps = 1e-8, q = None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.q = q

    def hybrid_forward(self, F, pred, label, mask):
        flow, flow_var = F.split(pred,axis=1,num_outputs=2)
        if self.q is not None:
            loss = (F.sum(F.abs(flow - label), axis = 1) + self.eps) ** self.q
        else:
            loss = F.sqrt(F.sum(F.square(flow - label), axis = 1) + self.eps)
        loss = F.broadcast_mul(loss, mask.squeeze(axis = 1))
        loss = F.sum(loss, axis=0, exclude=True) / F.sum(mask, axis=0, exclude=True)
        return loss
        


class MultiscaleEpeProb(nn.HybridBlock):
    def __init__(self, scales, weights, match, eps = 1e-8, q = None, **kwargs):
        super().__init__(**kwargs)
        
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q
        if match == 'upsampling':
            with self.name_scope():
                for s in self.scales:
                    setattr(self, 'upsampler_{}'.format(s), Upsample(s))

    def _get_upsampler(self, s):
        return getattr(self, 'upsampler_{}'.format(s))

    def hybrid_forward(self, F, flow, mask, *predictions):
        if self.match == 'upsampling':
            losses = [EpeLossWithMaskProb(eps = self.eps, q = self.q)(self._get_upsampler(s)(p), flow, mask) * w
                    for p, w, s in zip(predictions, self.weights, self.scales)]
        elif self.match == 'downsampling':
            losses = [EpeLossWithMaskProb(eps = self.eps, q = self.q)(p, Downsample(s)(flow), Downsample(s)(mask)) * w
                    for p, w, s in zip(predictions, self.weights, self.scales)]
        else:
            raise NotImplementedError
        return F.add_n(*losses)
        
class MultiscaleNllEpeProb(nn.HybridBlock):
    def __init__(self, scales, weights, match, eps = 1e-8, q = None, **kwargs):
        super().__init__(**kwargs)
        
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q
        if match == 'upsampling':
            with self.name_scope():
                for s in self.scales:
                    setattr(self, 'upsampler_{}'.format(s), Upsample(s))

    def _get_upsampler(self, s):
        return getattr(self, 'upsampler_{}'.format(s))

    def hybrid_forward(self, F, flow, mask, *predictions):
        if self.match == 'upsampling':
            losses = []
            for p, w, s in zip(predictions, self.weights, self.scales):
                l = NllLossWithMaskProb(eps = self.eps, q = self.q)(self._get_upsampler(s)(p), flow, mask) * w
                losses.append(l)
        elif self.match == 'downsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                l = NllLossWithMaskProb(eps = self.eps, q = self.q)(p, Downsample(s)(flow), Downsample(s)(mask)) * w
                losses.append(l)
        else:
            raise NotImplementedError
        return F.add_n(*losses)
        
class MultiscaleCalEpeProb(nn.HybridBlock):
    def __init__(self, scales, weights, match, eps = 1e-8, q = None, **kwargs):
        super().__init__(**kwargs)
        
        self.scales = scales
        self.weights = weights
        self.match = match
        self.eps = eps
        self.q = q
        if match == 'upsampling':
            with self.name_scope():
                for s in self.scales:
                    setattr(self, 'upsampler_{}'.format(s), Upsample(s))

    def _get_upsampler(self, s):
        return getattr(self, 'upsampler_{}'.format(s))

    def hybrid_forward(self, F, flow, mask, *predictions):
        if self.match == 'upsampling':
            losses = []
            for p, w, s in zip(predictions, self.weights, self.scales):
                l1 = NllLossWithMaskProb(eps = self.eps, q = self.q)(self._get_upsampler(s)(p), flow, mask) * w
                l2 = 1e-3 * WassersteinCalLossProb()(self._get_upsampler(s)(p), flow) * w
                l = F.broadcast_add(l1,l2)
                losses.append(l)
        elif self.match == 'downsampling':
            for p, w, s in zip(predictions, self.weights, self.scales):
                l1 = NllLossWithMaskProb(eps = self.eps, q = self.q)(p, Downsample(s)(flow), Downsample(s)(mask)) * w
                l2 = 1e-3 * WassersteinCalLossProb()(p, Downsample(s)(flow)) * w
                l = F.broadcast_add(l1,l2)
                losses.append(l)
        else:
            raise NotImplementedError
        return F.add_n(*losses)
