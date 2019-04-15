"""
Created on Apr 12, 2019
@author: Yuedong Chen
"""

from data import create_dataloader
from model import create_model
from visualizer import Visualizer
import copy
import time
import os
import torch
import numpy as np
from PIL import Image


def create_solver(opt):
    instance = Solver()
    instance.initialize(opt)
    return instance


class Solver(object):
    """docstring for Solver"""
    def __init__(self):
        super(Solver, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.visual = Visualizer()
        self.visual.initialize(self.opt)

    def run_solver(self):
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks()

    def train_networks(self):
        # init train setting
        self.init_train_setting()

        # for every epoch
        for epoch in range(self.opt.epoch_count, self.epoch_len + 1):
            # train network
            self.train_epoch(epoch)
            # update learning rate
            self.cur_lr = self.train_model.update_learning_rate()
            # save checkpoint if needed
            if epoch % self.opt.save_epoch_freq == 0:
                self.train_model.save_ckpt(epoch)

    def init_train_setting(self):
        self.train_dataset = create_dataloader(self.opt)
        self.train_model = create_model(self.opt)

        self.train_total_steps = 0
        self.epoch_len = self.opt.niter + self.opt.niter_decay
        self.cur_lr = self.opt.lr

    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        epoch_steps = 0

        last_print_step_t = time.time()
        for idx, batch in enumerate(self.train_dataset):

            self.train_total_steps += self.opt.batch_size
            epoch_steps += self.opt.batch_size
            # train network
            self.train_model.feed_batch(batch)
            self.train_model.optimize_paras(train_recog=(idx % self.opt.train_recog_iter == 0))
            # print losses
            if self.train_total_steps % self.opt.print_losses_freq == 0:
                cur_losses = self.train_model.get_latest_losses()
                avg_step_t = (time.time() - last_print_step_t) / self.opt.print_losses_freq
                last_print_step_t = time.time()
                # print loss info to command line
                info_dict = {'epoch': epoch, 'epoch_len': self.epoch_len,
                            'epoch_steps': idx * self.opt.batch_size, 'epoch_steps_len': len(self.train_dataset),
                            'step_time': avg_step_t, 'cur_lr': self.cur_lr,
                            'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file),
                            'losses': cur_losses
                            }
                self.visual.print_losses_info(info_dict)
            
            # plot loss map to visdom
            if self.train_total_steps % self.opt.plot_losses_freq == 0 and self.visual.display_id > 0:
                cur_losses = self.train_model.get_latest_losses()
                epoch_steps = idx * self.opt.batch_size
                self.visual.display_current_losses(epoch - 1, epoch_steps / len(self.train_dataset), cur_losses)

            # display image on visdom
            if self.train_total_steps % self.opt.sample_img_freq == 0 and self.visual.display_id > 0:
                cur_vis = self.train_model.get_latest_visuals()
                self.visual.display_online_results(cur_vis, epoch)                
            
    def test_networks(self):
        self.init_test_setting(self.opt)
        self.test_ops()

    def init_test_setting(self):
        self.test_dataset = create_dataloader(self.opt)
        self.test_model = create_model(self.opt)

    def test_ops(self):
        for batch_idx, batch in enumerate(self.test_dataset):
            with torch.no_grad():

                self.test_model.feed_batch(batch)
                self.test_model.forward()
        