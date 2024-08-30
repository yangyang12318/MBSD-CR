import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x

def normalize_imagenet_opt(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
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
        """
        if refer.size(1)==3:
            refer=refer
        else:
            refer=torch.stack((refer,refer,refer),dim=1)
            refer=torch.squeeze(refer,dim=2)


        if self.image_encoder.normalize:
            cl_tensor=normalize_imagenet(refer)

        else:
            cl_tensor = refer
        """
        if status =='train':
            """
            image_features = self.image_encoder.features.conv1(cl_tensor) #(2,32,128,128)
            image_features = self.image_encoder.features.bn1(image_features)
            image_features = self.image_encoder.features.act1(image_features)
            image_features = self.image_encoder.features.maxpool(image_features)
            image_features=self.image_encoder.features.layer1(image_features)#(2,192,64,64)
            image_embd_layer1=self.avgpool_cl(image_features)  #(2,192,8,8)

          
            sar_embd_layer1=self.avgpool_sar(sar)#(2,192,8,8)
            image_features_layer1, sar_features_layer1 = self.transformer1(image_embd_layer1, sar_embd_layer1, 1)
          
            sar_features_layer1 = F.interpolate(sar_features_layer1,
                                                  size=(sar.shape[2], sar.shape[3]), mode='bilinear',
                                                  align_corners=False)
            sar = sar + sar_features_layer1
            """
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

            """
            image_features = self.image_encoder.features.layer2(image_features)  # (2,384,64,64)
            image_features = self.image_encoder.features.layer3(image_features)  # (2,792,32,32)
            image_embd_layer2 = self.avgpool_cl(image_features)  # (2,192,8,8)


            sar_embd_layer2 = self.avgpool_sar(sar)  # (2,192,8,8)
            image_features_layer2, sar_features_layer2 = self.transformer2(image_embd_layer2, sar_embd_layer2, 1)
           
            sar_features_layer2 = F.interpolate(sar_features_layer2,
                                                size=(sar.shape[2], sar.shape[3]), mode='bilinear',
                                                align_corners=False)
           
            sar = sar + sar_features_layer2
            """
            #sar = self.tail(sar)
           # sar = self.end(sar)
            return sar
        else:
            """
            image_features = self.image_encoder.features.conv1(cl_tensor)  # (2,32,128,128)
            image_features = self.image_encoder.features.bn1(image_features)
            image_features = self.image_encoder.features.act1(image_features)
            image_features = self.image_encoder.features.maxpool(image_features)
            image_features = self.image_encoder.features.layer1(image_features)  # (2,192,64,64)
            image_embd_layer1 = self.avgpool_cl(image_features)  # (2,192,8,8)
            """

            """
            sar_embd_layer1 = self.avgpool_sar(sar)  # (2,192,8,8)
            image_features_layer1, sar_features_layer1 = self.transformer1(image_embd_layer1, sar_embd_layer1, 1)
            sar_features_layer1 = F.interpolate(sar_features_layer1,
                                                size=(sar.shape[2], sar.shape[3]), mode='bilinear',
                                                align_corners=False)
            # image_features = image_features + image_features_layer1
            sar = sar + sar_features_layer1
            """
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
            #sar=self.tail(sar)
            """
            image_features = self.image_encoder.features.layer2(image_features)  # (2,384,64,64)
            image_features = self.image_encoder.features.layer3(image_features)  # (2,792,32,32)
            image_embd_layer2 = self.avgpool_cl(image_features)  # (2,192,8,8)

            sar_embd_layer2 = self.avgpool_sar(sar)  # (2,192,8,8)
            image_features_layer2, sar_features_layer2 = self.transformer2(image_embd_layer2, sar_embd_layer2, 1)
          
            sar_features_layer2 = F.interpolate(sar_features_layer2,
                                                size=(sar.shape[2], sar.shape[3]), mode='bilinear',
                                                align_corners=False)
         
            sar = sar + sar_features_layer2
            """
            #sar = self.end(sar)
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


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 img_vert_anchors, img_horz_anchors,
                 lidar_vert_anchors, lidar_horz_anchors,
                 seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop,  use_velocity=True):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1

        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors


        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1,
                                                self.seq_len * img_vert_anchors * img_horz_anchors + self.seq_len * lidar_vert_anchors * lidar_horz_anchors,
                                                n_embd))

        # velocity embedding
        self.use_velocity = use_velocity
        if (use_velocity == True):
            self.vel_emb = nn.Linear(self.seq_len, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,
                                       std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0, 1, 3, 4, 2).contiguous().view(
            bz, -1, self.n_embd) #self.n_embd就是h*w
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w).permute(0, 1, 3, 4,
                                                                                         2).contiguous().view(bz, -1,
                                                                                                              self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        # project velocity to n_embed
        if (self.use_velocity == True):
            velocity_embeddings = self.vel_emb(velocity)  # (B, C)
            # add (learnable) positional embedding and velocity embedding for all tokens
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1))  # (B, an * T, C)
        else:
            x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        x = x.view(bz,
                   self.seq_len * self.img_vert_anchors * self.img_horz_anchors + self.seq_len * self.lidar_vert_anchors * self.lidar_horz_anchors,
                   self.n_embd)

        image_tensor_out = x[:, :self.seq_len * self.img_vert_anchors * self.img_horz_anchors, :].contiguous().view(
            bz * self.seq_len, -1, img_h, img_w)
        lidar_tensor_out = x[:, self.seq_len * self.img_vert_anchors * self.img_horz_anchors:, :].contiguous().view(
            bz * self.seq_len, -1, lidar_h, lidar_w)

        return image_tensor_out, lidar_tensor_out

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

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


class Coupled_Layer(nn.Module):
    def __init__(self,
                 coupled_number=32,
                 n_feats=64,
                 kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        self.coupled_number = coupled_number
        self.kernel_size = kernel_size
        # kernel_shared_1是一个（32，64，3，3）并且初始化后的卷积核，32是输出通道数，64是输入通道数，3，3是卷积核的高度和宽度
        self.kernel_shared_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.zeros(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_1 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_shared_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_depth_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))
        self.kernel_rgb_2 = nn.Parameter(nn.init.kaiming_uniform(
            torch.randn(size=[self.n_feats - self.coupled_number, self.n_feats, self.kernel_size, self.kernel_size])))

        self.bias_shared_1 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_1 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

        self.bias_shared_2 = nn.Parameter((torch.zeros(size=[self.coupled_number])))
        self.bias_depth_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))
        self.bias_rgb_2 = nn.Parameter((torch.zeros(size=[self.n_feats - self.coupled_number])))

    def forward(self, feat_dlr, feat_rgb):
        # feat_dlr feat_rgb是经过初始化以后的深度图像和rgb图像特征
        shortCut = feat_dlr
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_1, self.kernel_depth_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_depth_1], dim=0),
                            padding=1)
        feat_dlr = F.relu(feat_dlr, inplace=True)
        feat_dlr = F.conv2d(feat_dlr,
                            torch.cat([self.kernel_shared_2, self.kernel_depth_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_depth_2], dim=0),
                            padding=1)
        feat_dlr = F.relu(feat_dlr + shortCut, inplace=True)
        shortCut = feat_rgb
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_1, self.kernel_rgb_1], dim=0),
                            torch.cat([self.bias_shared_1, self.bias_rgb_1], dim=0),
                            padding=1)
        feat_rgb = F.relu(feat_rgb, inplace=True)
        feat_rgb = F.conv2d(feat_rgb,
                            torch.cat([self.kernel_shared_2, self.kernel_rgb_2], dim=0),
                            torch.cat([self.bias_shared_2, self.bias_rgb_2], dim=0),
                            padding=1)
        feat_rgb = F.relu(feat_rgb + shortCut, inplace=True)
        return feat_dlr, feat_rgb