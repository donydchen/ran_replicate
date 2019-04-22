import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class BaseDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.data_root.strip('/'))

    def initialize(self, opt):
        self.opt = opt
        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir)
        self.is_train = self.opt.mode == "train"

        # load images path 
        filename = self.opt.train_csv if self.is_train else self.opt.test_csv
        self.imgs_name_file = os.path.join(self.opt.data_root, filename)
        self.imgs_path = self.load_image(self.imgs_dir, self.imgs_name_file)

        # load pseudo expression and AU label 
        pseudo_csv = os.path.join(self.opt.data_root, self.opt.pseudo_csv)
        self.pseudo_list = self.load_csv(pseudo_csv)

        # load images expression label dicitionary 
        cls_pkl = os.path.join(self.opt.data_root, self.opt.cls_pkl)
        self.cls_dict = self.load_dict(cls_pkl)

        # load AUs vector in testing 
        if not self.is_train:
            aus_pkl = os.path.join(self.opt.data_root, self.opt.aus_pkl)
            self.aus_dict = self.load_dict(aus_pkl)

    def load_image(self, imgs_dir, imgs_name_file):
        return None

    def load_csv(self, csv_path):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def get_cls_by_path(self, img_path):
        img_name = os.path.basename(img_path)
        assert img_name in self.cls_dict, "Cannot find label for %s" % img_name
        cls_label = self.cls_dict[img_name]
        return cls_label

    def get_aus_by_path(self, img_path):
        img_name = os.path.basename(img_path)
        assert img_name in self.aus_dict, "Cannot find AUs vector for %s" % img_name
        aus_label = self.aus_dict[img_name]
        # print("aus_label:", aus_label, type(aus_label))
        return aus_label

    def img_transform(self, img, no_data_augment=False):
        img2tensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        img = transforms.functional.resize(img, self.opt.load_size)
        # on-the-fly data augmentation
        if self.opt.mode == "train" and not no_data_augment:
            # scale and crop 
            # lucky_num = random.randint(0, 4)
            lucky_num_crop = random.randint(0, 4)
            img = transforms.functional.five_crop(img, self.opt.final_size)[lucky_num_crop]
            # Horizontally flip
            lucky_num_flip = random.randint(0, 1)
            if lucky_num_flip:
                img = transforms.functional.hflip(img)
        else:
            img = transforms.functional.five_crop(img, self.opt.final_size)[-1]  # center crop

        return img2tensor(img)

    def __len__(self):
        return len(self.imgs_path)












