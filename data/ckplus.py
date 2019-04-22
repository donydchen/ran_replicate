from .base_dataset import BaseDataset
import os
import numpy as np


class CKPlusDataset(BaseDataset):
    """docstring for CKPlusDataset"""
    def __init__(self):
        super(CKPlusDataset, self).__init__()

    def name(self):
        return "CKPlusDataset"

    def initialize(self, opt):
        self.EXPRESSION = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
        super(CKPlusDataset, self).initialize(opt)
        
    def load_image(self, imgs_dir, imgs_name_file):
        imgs = []
        assert os.path.isfile(imgs_name_file), "File '%s' does not exist." % imgs_name_file
        with open(imgs_name_file, 'r') as f:
            lines = f.readlines()
            imgs = [os.path.join(imgs_dir, line.strip()) for line in lines]
            imgs = sorted(imgs)
        return imgs

    def load_csv(self, csv_path):
        pseudo_list = []
        assert os.path.isfile(csv_path), "File '%s' does not exist." % csv_path
        with open(csv_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue  # skip the header 
                cur_list = []
                items = line.strip().split(',')
                cur_list.append(self.EXPRESSION.index(items[0]))
                cur_list.append(list(map(lambda x: float(x), items[1:])))
                pseudo_list.append(cur_list)
        return pseudo_list

    def get_cls_by_path(self, img_path):
        cls_label = super(CKPlusDataset, self).get_cls_by_path(img_path)
        return self.EXPRESSION.index(cls_label)

    def __getitem__(self, index):
        # load pseudo expression and its AU label
        pseudo_exp = int(self.pseudo_list[index][0])
        pseudo_aus = np.array(self.pseudo_list[index][1])

        # load image and its expression label 
        img_path = self.imgs_path[index]
        img_exp = int(self.get_cls_by_path(img_path))
        img_tensor = self.img_transform(self.get_img_by_path(img_path), self.opt.no_data_augment)

        data_dict = {'pseudo_aus': pseudo_aus, 'pseudo_exp': pseudo_exp,
                     'img_exp': img_exp, 'img': img_tensor}

        # load image AUs in testing
        if not self.is_train:
            img_aus = np.array(self.get_aus_by_path(img_path))
            # print("img_aus", img_aus, type(img_aus))
            data_dict['img_aus'] = img_aus

        return data_dict



