# -*- coding: utf-8 -*-
import os
import scipy.io as sio
from PIL import Image
import numpy as np
import pickle
import random

CWD = os.getcwd()


class ReadData:
    """
    This class is pre-processing a folder of images and corresponding digitStruct.mat file.
    The original train.tar.gz and test.tar.gz have been downloaded and unzip.
    Also, I randomly select 4168 images from training set to form validation set.
    """

    def __init__(self, cwd='/Users/haoran/Desktop/train/'):
        """
        initiate class container for images and labels
        :param cwd: the absolute path for the folder where images and digitStruct.mat store
        """
        self.image_lst = []
        self.label_data = {}
        self.root_dir = cwd

    def get_image_lst(self):
        """
        get all the names of images in the given path
        """
        f_lst = os.listdir(os.getcwd())
        rm_list = []
        for item in f_lst:
            if item[-3:] != 'png':
                rm_list.append(item)
        for item in rm_list:
            f_lst.remove(item)
        self.image_lst = f_lst

    def load_data(self):
        """
        This function is used for reading data, such as labels and positions in a image, from digitstruct2.mat file.
        Then, store them in self.label_data.
        """
        data = {}
        try:
            mat = sio.loadmat('digitStruct2.mat')
            digitStruct = mat['digitStruct'][0]
        except Exception as e:
            print "error when load labels: ", e

        for item in digitStruct:
            name = item[0][0]
            bbox = item[1][0]
            bbox_lst = []
            for item2 in bbox:
                # check validation:
                label = item2[4][0][0]
                if label < 1 or label > 10:
                    print "abnormal label: ", label
                sub_dic = {}
                sub_dic.setdefault('height', item2[0][0][0])
                sub_dic.setdefault('left', item2[1][0][0])
                sub_dic.setdefault('top', item2[2][0][0])
                sub_dic.setdefault('width', item2[3][0][0])
                sub_dic.setdefault('label', item2[4][0][0])
                bbox_lst.append(sub_dic)
            data.setdefault(name, bbox_lst)

        self.label_data = data

    def fill_feed_dict(self):
        """
        Load label data, and use the position information to crop original images,
        then resize them into uniform 32*32 size, grey scale

        :return: result_dict: containing an array of cropped labels, an array of labels in one-hotting form,
        and a list of names of all the images
        """
        os.chdir(self.root_dir)
        self.get_image_lst()
        self.load_data()

        # shuffle the image_lst in preprocess
        random.shuffle(self.image_lst)

        images_feed = []
        labels_feed = []
        for item in self.image_lst:

            # get labels from self.label_data and save them in list
            labels = []
            position = {'top': [], 'left': [], 'bottom': [], 'right': []}

            for idx in range(5):
                a_digit = [0] * 11  # ONE-HOT ENCODING
                if len(self.label_data[
                           item]) <= idx:  # if the length of the label <= the current index, this index is empty
                    a_digit[-1] += 1.0
                    labels.append(a_digit)
                else:
                    label = self.label_data[item][idx]['label']
                    a_digit[label % 10] += 1.0
                    labels.append(a_digit)
                    position['top'].append(self.label_data[item][idx]['top'])  # record the positions
                    position['left'].append(self.label_data[item][idx]['left'])
                    bottom = (self.label_data[item][idx]['top'] + self.label_data[item][idx]['height'])
                    position['bottom'].append(bottom)
                    right = (self.label_data[item][idx]['left'] + self.label_data[item][idx]['width'])
                    position['right'].append(right)
            labels_feed.append(labels)

            # Crop the image, then convert the image into 32*32 uniform size
            crop_lst = [int(min(position['left'])), int(min(position['top'])), int(max(position['right'])),
                        int(max(position['bottom']))]    # determine the coordinate to crop
            img = Image.open(item)
            img = img.crop(crop_lst).convert('L').resize((32, 32))  # set a uniform image size
            img_ndarray = np.asarray(img, dtype='float32')
            img_lst = img_ndarray.reshape(1, img_ndarray.size) / 256
            images_feed.append(img_lst.tolist()[0])

        result_dict = {         # pack the information into result_dict, then return it
            'images': np.array(images_feed, dtype="float32"),
            'labels': np.array(labels_feed, dtype="float32"),
            'name': self.image_lst
        }

        return result_dict


def main():
    """
    main function to set path and start the pre-processing
    """
    train = ReadData()
    train_dic = train.fill_feed_dict()
    pickle.dump(train_dic, open(CWD + '/train_data.p', 'wb'), -1)

    va = ReadData('/Users/haoran/Desktop/validation/')
    va_dic = va.fill_feed_dict()
    pickle.dump(va_dic, open(CWD + '/va_data.p', 'wb'), -1)

    test = ReadData('/Users/haoran/Desktop/test/')
    test_dic = test.fill_feed_dict()
    pickle.dump(test_dic, open(CWD + '/test_data.p', 'wb'), -1)


main()
