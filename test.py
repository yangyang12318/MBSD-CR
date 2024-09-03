
import torch.nn as nn
from torch.optim import lr_scheduler
from models.DBSNL import DBSNl
from torch.utils.data import DataLoader

from util.file_manager import FileManager
from util.logger import Logger
import time, datetime
from datahandler.SAR import SARdatatest
import argparse, os

import torch


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name',   type=str)
    args.add_argument('-e', '--ckpt_epoch',   default=20,     type=int)
    args.add_argument( '--gpu',          default='0',  type=str)
    args.add_argument(      '--thread',       default=32,     type=int)


    # 模型参数
    args.add_argument('--in_ch', type=int, default=1)  # 输入模型通道数
    args.add_argument('--out_ch', type=int, default=1)  # 模型输出通道数
    args.add_argument('--base_ch', type=int, default=192)  # 第一层卷积转化后的特征图通道数
    args.add_argument('--num_module', type=int, default=9)  # 中间特征提取过程迭代次数，

    #数据参数
    # 评估数据参数
    args.add_argument('--test_data', default=SARdatatest)
    args.add_argument('--test_add_noise', default=None)
    args.add_argument('--test_crop_size', default=None)


    args = args.parse_args()
    file_manager = FileManager(args.session_name)
    logger = Logger()
    # device setting
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  #定义模型
    network = nn.DataParallel(DBSNl(args.in_ch, args.out_ch, args.base_ch, args.num_module)).cuda()
    optimizer = torch.optim.Adam(network.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)


    # 加载训练数据
    testdata = args.test_data(add_noise=args.test_add_noise, crop_size=args.test_crop_size)
    test_dataloader = DataLoader(dataset=testdata, batch_size=1, shuffle=False, num_workers=args.thread)
    file_manager._set_status('test')
    ckpt_epoch=file_manager._find_last_epoch()  if args.ckpt_epoch == -1 else args.ckpt_epoch
    checkpoint_path = os.path.join(file_manager.get_dir('checkpoint'), f'checkpoint_epoch_{ckpt_epoch}.pth')
    testcheckpoint = torch.load(checkpoint_path, map_location='cuda')
    network.module.load_state_dict(testcheckpoint['model_weight'])
    optimizer.load_state_dict(testcheckpoint['optimizer_weight'])
    scheduler.load_state_dict(testcheckpoint['scheduler_state'])
    network.eval()
    file_manager._set_status('test %03d' % ckpt_epoch)
    logger.highlight(logger.get_start_msg())
    denoiser = network.module

    # set image save path

    test_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
    img_save_path = 'img/test_%s_%03d_%s' % (str(args.session_name), ckpt_epoch, test_time)
    file_manager.make_dir(img_save_path)
    for idx, data in enumerate(test_dataloader):
        if args.gpu is not None:
            for key in data:
                if key != 'file_name':
                    data[key] = data[key].to('cuda')
        file_name = (data['file_name'][0][:-4])
        test_result = denoiser(data['real_noisy'],data['NLsar'],'test')
        denoi_name = '%s_DN' % file_name
        test_result = test_result.squeeze(0).cpu()
        file_manager.save_img_tensor(img_save_path, denoi_name, test_result)

        if 'clean' in data:
            logger.note('[%s] testing... %04s/%04s. ' % (
                file_manager.status, file_name, test_dataloader.__len__()), end='\r')
        else:
            logger.note('[%s] testing... %04s/%04s.' % (file_manager.status, file_name, test_dataloader.__len__()),
                        end='\r')

if __name__ == '__main__':
    main()


