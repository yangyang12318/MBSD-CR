import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from .util import tensor2np, human_format

status_len = 13
class FileManager():
    def __init__(self, session_name:str):
        self.output_folder = "/home/peter/yy/mount_a5000/mycode3/output"
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)
            print("[WARNING] output folder is not exist, create new one")

        # init session
        self.session_name = session_name
        os.makedirs(os.path.join(self.output_folder, self.session_name), exist_ok=True)

        # mkdir
        for directory in ['checkpoint', 'img', 'tboard']:
            self.make_dir(directory)

        self.checkpoint_folder='checkpoint'

    def is_dir_exist(self, dir_name:str) -> bool:
        return os.path.isdir(os.path.join(self.output_folder, self.session_name, dir_name))

    def make_dir(self, dir_name:str) -> str:
        os.makedirs(os.path.join(self.output_folder, self.session_name, dir_name), exist_ok=True) 

    def get_dir(self, dir_name:str) -> str:
        # -> './output/<session_name>/dir_name'
        return os.path.join(self.output_folder, self.session_name, dir_name)

    def save_img_tensor(self, dir_name:str, file_name:str, img:torch.Tensor, ext='png'):
        self.save_img_numpy(dir_name, file_name, tensor2np(img), ext)

    def save_img_numpy(self, dir_name:str, file_name:str, img:np.array, ext='png'):
        file_dir_name = os.path.join(self.get_dir(dir_name), '%s.%s'%(file_name, ext))
        if np.shape(img)[2] == 1:
            cv2.imwrite(file_dir_name, np.squeeze(img, 2))
        else:
            cv2.imwrite(file_dir_name, img)

    def _find_last_epoch(self):

        checkpoint_list = os.listdir(self.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('checkpoint_epoch_', '').replace('.pth', '')) for ckpt in checkpoint_list]
        print(max(epochs))
        return max(epochs)

    def load_checkpoint(self, load_epoch=0, name=None):
        if name is None:
            # if scratch, return
            if load_epoch == 0: return
            # load from local checkpoint folder
            file_name = os.path.join(self.get_dir(self.checkpoint_folder),
                                     self._checkpoint_name(load_epoch))
        else:
            # load from global checkpoint folder
            file_name = os.path.join('./ckpt', name)

        # check file exist
        assert os.path.isfile(file_name), 'there is no checkpoint: %s' % file_name

        # load checkpoint (epoch, model_weight, optimizer_weight)
        saved_checkpoint = torch.load(file_name)


        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.optimizer:
                self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

        # print message
        # self.logger.note('[%s] model loaded : %s'%(self.status, file_name))

    def _checkpoint_name(self,epoch):
        return self.session_name +'_%03d'%epoch +'.pth'

    def summary(self,network):
        summary = ''

        summary += '-' * 100 + '\n' #写日志的第一行，就是

        # Check if the model is wrapped in nn.DataParallel
        if isinstance(network, nn.DataParallel):
            network = network.module
        # get parameter number
        param_num = sum(p.numel() for p in network.parameters() if p.requires_grad)

        # get information about architecture and parameter number
        summary += '[%s] parameters: %s -->\n' % (network.__class__.__name__, human_format(param_num))
        summary += str(network) + '\n\n'

        # optim

        # Hardware

        summary += '-' * 100 + '\n'

        return summary

    def _set_status(self, status: str):
        assert len(status) <= status_len, 'status string cannot exceed %d characters, (now %d)' % (
            status_len, len(status))

        if len(status.split(' ')) == 2:
            s0, s1 = status.split(' ')
            self.status = '%s' % s0.rjust(status_len // 2) + ' ' \
                                                             '%s' % s1.ljust(status_len // 2)
        else:
            sp = status_len - len(status)
            self.status = ''.ljust(sp // 2) + status + ''.ljust((sp + 1) // 2)




    