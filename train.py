import argparse, os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import timm

from Loss.drection_loss import gradient_direction_loss
from datahandler.SAR import SARdata,SARdatatest,SARdataval
from models.DBSNL import DBSNl
from models.APUNet import APUNet
from torch.utils.data import DataLoader
import math
from util.file_manager import FileManager
from util.logger import Logger
import time, datetime
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from Loss import Loss
import numpy as np

import torch

def total_variation(image_in):

    tv_h = torch.sum(torch.abs(image_in[ :, :-1] - image_in[ :, 1:]))
    tv_w = torch.sum(torch.abs(image_in[ :-1, :] - image_in[ 1:, :]))
    tv_loss = tv_h + tv_w

    return tv_loss

def TV_loss(im_batch, weight):
    TV_L = 0.0

    for tv_idx in range(len(im_batch)):
        TV_L = TV_L + total_variation(im_batch[tv_idx,0,:,:])

    TV_L = TV_L/len(im_batch)

    return weight*TV_L


parser=argparse.ArgumentParser()
#全局参数
parser.add_argument('--resume',  default=False,action='store_true')
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)
parser.add_argument('--thread',    default=32,     type=int)
parser.add_argument('--epoch',    default=20,     type=int)
parser.add_argument('--session_name', default='SAR_hunan',type=str)
parser.add_argument('--warmup', default=False,type=bool)

#训练数据参数
parser.add_argument('--train_data', default=SARdata)
parser.add_argument('--train_add_noise',  default=None)
parser.add_argument('--train_crop_size',  default=[250, 250])
parser.add_argument('--train_aug',  default=['hflip', 'rot'])
parser.add_argument('--train_n_repeat',  default=8)


#评估数据参数
parser.add_argument('--val_data', default=SARdataval)
parser.add_argument('--val_add_noise',  default=None)
parser.add_argument('--val_crop_size',  default=None)
parser.add_argument('--val_n_data',  default=50)

#模型参数
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--in_ch', type=int, default=1)  #输入模型通道数
parser.add_argument('--out_ch', type=int, default=1) #模型输出通道数
parser.add_argument('--base_ch', type=int, default=192) #第一层卷积转化后的特征图通道数
parser.add_argument('--num_module',type=int,default=9)  #中间特征提取过程迭代次数，
parser.add_argument('--lr',type=int,default=1e-4)  #初始学习率设置
args=parser.parse_args()

def main():
   # print([m for m in timm.list_models() if m.startswith('convnext')])
    #各种包管理函数
    file_manager = FileManager(args.session_name)


    # device setting
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

     # cudnn
    torch.backends.cudnn.benchmark = False


    network=nn.DataParallel(DBSNl(args.in_ch, args.out_ch,args.base_ch,args.num_module)).cuda()


    #加载训练数据
    traindata=args.train_data(add_noise=args.train_add_noise,crop_size=args.train_crop_size,aug= args.train_aug,n_repeat= args.train_n_repeat)
    train_dataloader=DataLoader(dataset=traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.thread)

    # 评估数据加载
    if args.val_data is not None:
        valdata = args.val_data(add_noise=args.val_add_noise, crop_size=args.val_crop_size, n_data=args.val_n_data)
        val_dataloader = DataLoader(dataset=valdata, batch_size=1, shuffle=False, num_workers=args.thread)

    #设置epoch
    max_epoch=args.epoch
    #起始epoch设置，这个应该不是多余的设置，因为后面如果resume的跑模型的话，可能这个start_epoch不是1
    epoch=start_epoch=1
    #max_len是图片的总数，因为后面要可视化迭代训练的过程， 这个max_iter是总共的图片数量除以设置的batch size 然后向上取整 就是一共需要迭代的次数
    max_len=train_dataloader.dataset.__len__()
    max_iter=math.ceil(max_len / args.batch_size)

    #设置loss
    self_loss=Loss('1*L1')
    loss_dict={'count':0}
    loss_log = []

    #设置 optimizer
    optimizer = torch.optim.Adam(network.parameters(), args.lr,betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)


    if args.resume:
        # find last checkpoint
        load_epoch = file_manager._find_last_epoch()

        # load last checkpoint
        checkpoint_path = os.path.join(file_manager.get_dir('checkpoint'), f'checkpoint_epoch_{load_epoch}.pth')
        testcheckpoint = torch.load(checkpoint_path, map_location='cuda')
        network.module.load_state_dict(testcheckpoint['model_weight'])
        optimizer.load_state_dict(testcheckpoint['optimizer_weight'])
        scheduler.load_state_dict(testcheckpoint['scheduler_state'])
        # file_manager.load_checkpoint(load_epoch)
        epoch = load_epoch + 1

        # logger initialization  这里就是初始化一下logger，然后后面填写
        logger = Logger((max_epoch, max_iter), log_dir=file_manager.get_dir(''),
                        log_file_option='a')
    else:
        # logger initialization 这里就是初始化一下logger，然后后面填写
        logger = Logger((max_epoch, max_iter), log_dir=file_manager.get_dir(''),
                        log_file_option='w')  # 初始化logger的时候，就会创建一个在/output/session_name下面的一个log_日期的log文件，这个就是日志文件的位置 同时也会创建一个validation_日期的评估log文件
    # tensorboard  这个是设置tensorboard的
    tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
    tboard = SummaryWriter(log_dir=file_manager.get_dir('tboard/%s' % tboard_time))


    #开始写日志
    logger.info(file_manager.summary(network))
    logger.start((epoch-1,0))
    logger.highlight(logger.get_start_msg())  #


    #****************上面是before_train的定义部分************
    for epoch in range(epoch,max_epoch+1):
        file_manager._set_status('epoch %03d/%03d' % (epoch, max_epoch))  #这个就是把日志对齐一下，变成【epoch  001/020】这个样子，返回的是self，status这个玩意儿，我不知道
        # 每个epoch开始时重新创建数据加载器的迭代器
        train_dataloader_iter = iter(train_dataloader)

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
#        print("LearningRate of Epoch {} = {}".format(epoch, current_lr))
        network.train()

        for iteration in range(1, max_iter + 1):
            data = next(train_dataloader_iter)

            for key in data:
                if key != 'file_name':
                    data[key] = data[key].to('cuda')

            sar = network((data['real_noisy']),data['NLsar'] ,'train')


            tv_loss = TV_loss(sar, 0.000002)  # 中间的损失函数


            losses = self_loss(sar,  data)  # 最后的损失函数
            losses = {key: losses[key].mean() for key in losses}
            total_loss = sum(v for v in losses.values())

            optimizer.zero_grad()
            total_loss=total_loss+tv_loss
            total_loss.backward()
            optimizer.step()


            for key in losses:
                if key != 'count':
                    if key in loss_dict:
                        loss_dict[key] += float(losses[key])
                    else:
                        loss_dict[key] = float(losses[key])
            loss_dict['count']+=1


            if (iteration%10==0 and iteration!=0)  or iteration ==max_iter:
                temporal_loss = 0.
                for key in loss_dict:
                    if key != 'count':
                        temporal_loss += loss_dict[key] / loss_dict['count']
                loss_log += [temporal_loss]
                if len(loss_log) > 100: loss_log.pop(0)

                # print status and learning rate
                loss_out_str = '[%s] %04d/%04d, lr:%s ∣ ' % (
                    file_manager.status, iteration, max_iter, "{:.1e}".format(current_lr))
                global_iter = (epoch - 1) * max_iter + iteration

                # print losses
                avg_loss = np.mean(loss_log)
                loss_out_str += 'avg_100 : %.3f ∣ ' % (avg_loss)
                tboard.add_scalar('loss/avg_100', avg_loss, global_iter)

                for key in loss_dict:
                    if key != 'count':
                        loss = loss_dict[key] / loss_dict['count']
                        loss_out_str += '%s : %.3f ∣ ' % (key, loss)
                        tboard.add_scalar('loss/%s' % key, loss, global_iter)
                        loss_dict[key] = 0.

                # reset
                loss_dict['count'] = 0
                logger.info(loss_out_str)

            logger.print_prog_msg((epoch - 1, iteration - 1))
        scheduler.step()



        # 保存模型状态，包括epoch, model weights, optimizer weights, 以及scheduler的状态
        state = {
            'epoch': epoch,
            'model_weight': network.module.state_dict(),  # 保存模型权重
            'optimizer_weight': optimizer.state_dict(),  # 保存优化器状态
            'scheduler_state': scheduler.state_dict()  # 保存scheduler状态
        }
        # 构造保存路径
        checkpoint_path = os.path.join(file_manager.get_dir('checkpoint'), f'checkpoint_epoch_{epoch}.pth')
        # 执行保存操作
        torch.save(state, checkpoint_path)

        if args.val_data is not None:

            with torch.no_grad():
                network.eval()
                file_manager._set_status('val %03d' % epoch)

                denoiser=network.module
                #valcheckpoint = torch.load(checkpoint_path, map_location='cuda')
                #network.module.load_state_dict(valcheckpoint['model_weight'])  # 更新 network.module 的状态
                # make directories for image saving
                img_save_path = 'img/val_%03d' % epoch
                file_manager.make_dir(img_save_path)
                count = 0
                #validation
                for idx, data in enumerate(val_dataloader):
                    #val_clean_noise = add_gamma_noise_to_grayscale_batch(data['clean'])
                    if args.gpu is not None:
                        for key in data:
                            if key != 'file_name':
                                data[key] = data[key].to('cuda')
                    file_name = (data['file_name'][0][:-4])
                   # val_clean_noise=val_clean_noise.to('cuda')
                    output_sar = denoiser((data['real_noisy']),data['NLsar'],'val')

                    output_sar = torch.floor(output_sar)
                    val_result=output_sar

                    count += 1

                    denoi_name = '%s_DN' % file_name
                    val_result = val_result.squeeze(0).cpu()
                    file_manager.save_img_tensor(img_save_path, denoi_name, val_result)

                    logger.note('[%s] testing... %04s/%04s.' % (file_manager.status, file_name, val_dataloader.__len__()),end='\r')

    logger.highlight(logger.get_finish_msg())


if __name__ == '__main__':
    main()