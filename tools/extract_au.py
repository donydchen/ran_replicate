"""
Created on Apr 15, 2019
@author: Yuedong Chen
"""
import os
import subprocess
import inspect
import time
import glob
import argparse
import csv
import numpy as np
import pickle


class AUSDetector(object):
    """Using OpenFace to detect Action Units"""
    def __init__(self):
        super(AUSDetector, self).__init__()
        self.ALL_AUS = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', \
                        'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU28', 'AU45']

    def initialize(self, opt):
        self.bin_path = opt.bin_path
        self.img_ext = opt.img_ext
        self.raw_img_dir = opt.raw_img_dir

        self.out_dir = os.path.join(opt.root_dir, 'aus_csv')
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        self.pkl_path = os.path.join(opt.root_dir, 'aus_openface.pkl')
        
        self.FILTER_AUS = sorted(map(lambda x: 'AU%02d' % int(x), list(opt.aus.split(','))))

    def run(self):
        total_aus_dict = {}

        imgs_path = self.get_image_list()  # [:3]
        imgs_len = len(imgs_path)
        total_cost = 0.0

        for idx, img_path in enumerate(imgs_path):
            start_t = time.time()
            total_aus_dict.update(self.detect_aus(img_path))

            cur_cost = time.time() - start_t
            total_cost += cur_cost
            avg_cost = total_cost / (idx + 1.)

            print("[Success][%d/%d] Got AU of %s in %.2fs, remaining %.2f mins." % (idx, imgs_len - 1,
                    os.path.basename(img_path), cur_cost, (imgs_len - idx - 1.) * avg_cost / 60.))

        with open(self.pkl_path, 'wb') as f:
            pickle.dump(total_aus_dict, f, protocol=2)

        return total_aus_dict

    def detect_aus(self, img_path):
        # run au bin
        img_name = os.path.basename(img_path)
        out_name = os.path.splitext(img_name)[0]
        with open(os.devnull, 'w') as shutup:
            command_list = [self.bin_path, '-f', img_path, '-out_dir', self.out_dir, '-of', out_name, '-aus']
            return_code = subprocess.call(command_list, stdout=shutup, stderr=shutup)

        # parse au
        csv_path = os.path.join(self.out_dir, out_name + ".csv")
        aus_dict = {}
        try:
            with open(csv_path, 'r') as f:
                csv_reader = csv.reader(f)
                for idx, row in enumerate(csv_reader):
                    # if idx == 0:
                    #     print(row[2:2+17])
                    if idx > 0:
                        aus_dict[img_name] = [int(float(row[19+i])) for i, n in enumerate(self.ALL_AUS) if n in self.FILTER_AUS]
        except IOError:
            with open(os.path.join(self.out_dir, 'err.log'), 'a+') as f:
                f.write("Fail to detect au on %s.\n" % img_name)

        # clear tmp file
        txt_path = os.path.join(self.out_dir, out_name + "_of_details.txt")
        if os.path.isfile(txt_path):
            os.remove(txt_path)

        # return current au list
        return aus_dict

    def get_image_list(self):
        # copy from preprocess_ckplus.py
        image_list = []
        for subject in glob.glob(os.path.join(self.raw_img_dir, '*/')):
            for clip in glob.glob(os.path.join(subject, '*/')):
                items = sorted(glob.glob(os.path.join(clip, '*.%s' % self.img_ext)))
                image_list.extend(items[-3:])
        print(len(image_list))
        return image_list


def main():
    ausDetector = AUSDetector()
    cur_file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bin_path', required=True, help="OpenFace binary path 'FaceLandmarkImg'.")
    parser.add_argument('--aus', type=str, default='1,2,4,5,6,7,9,12,17,23,25', help='AUs vector index.')
    parser.add_argument('--img_ext', type=str, default='png', help='Image extension.')
    parser.add_argument('--raw_img_dir', type=str, default=os.path.join(cur_file_path, '../datasets/CKPlus/RAW/cohn-kanade-images'), help='raw image dataset dir.')
    parser.add_argument('--root_dir', type=str, default=os.path.join(cur_file_path, '../datasets/CKPlus'), help='dataset root dir.')
    
    opt = parser.parse_args()

    ausDetector.initialize(opt)
    ausDetector.run()


if __name__ == "__main__":
    main()
