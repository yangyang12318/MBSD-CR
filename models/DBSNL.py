import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple


def normalize_imagenet_opt(x):

    x = x.clone()
    x=x/255.0
    return x

class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included.
    see our supple for more details.
    '''

    def __init__(self, in_ch=1, out_ch=1, base_ch=192, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"



        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head= nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch12 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        self.branch22 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [nn.Conv2d(base_ch * 4, base_ch*2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch//2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch//2, 1, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.tail = nn.Sequential(*ly)


    def forward(self, sar,refer,status):

        if status =='train':

            sar=pixel_shuffle_down_sampling(sar,5,2)
            refer=pixel_shuffle_down_sampling(refer,5,2)
            sar=self.head(sar)
            refer=self.head(refer)
            br1 = self.branch1(sar)
            br11=self.branch12(refer)
            br2 = self.branch2(sar)
            br22=self.branch22(refer)
            sar = torch.cat([br1,br11, br2,br22], dim=1)

            sar=pixel_shuffle_up_sampling(self.tail(sar), 5,2)


            return sar
        else:

            sar = pixel_shuffle_down_sampling(sar, 1, 2)
            refer = pixel_shuffle_down_sampling(refer, 1, 2)
            sar = self.head(sar)
            refer = self.head(refer)
            br1 = self.branch1(sar)
            br11 = self.branch12(refer)
            br2 = self.branch2(sar)
            br22 = self.branch22(refer)
            sar = torch.cat([br1, br11, br2, br22], dim=1)

            sar = pixel_shuffle_up_sampling(self.tail(sar), 1, 2)

            return sar


    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1, dilation=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)

class DC_branchlsar(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1, dilation=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

