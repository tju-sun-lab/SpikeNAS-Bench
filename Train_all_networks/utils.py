import os
import pickle
import time
from typing import Optional, Callable, Tuple, Any

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import re
import sys




class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def trans10to5(n, x):
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']
    b = []
    c = []
    while True:
        s = n // x
        y = n % x
        b = b + [y]
        if s == 0:
            break
        n = s
    for i in b[::-1]:
        c.append(a[i])
    return c


def get_cntmat(id):
    count_vector = np.zeros((1, 6)).ravel()
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    index = trans10to5(id, 5)

    flag2 = 1
    flag3 = 1

    for j in range(len(index)):
        count_vector[6 - len(index) + j] = index[j]

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = count_vector[5 - num]

    neigh2_cnts = con_mat @ con_mat
    neigh3_cnts = neigh2_cnts @ con_mat
    neigh4_cnts = neigh3_cnts @ con_mat
    connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

    for node in range(3):
        if connection_graph[node, 3] == 0:  # if any node doesnt send information to the last layer, remove it
            flag2 = 0
    if flag2 == 0: return -1

    for node in range(3):
        if connection_graph[0, node + 1] == 0:  # if any node doesnt get information from the input layer, remove it
            flag3 = 0
        if flag3 == 0: return -1

    # 保证node1到node4一定有连接
    if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

    return con_mat

def get_cntmat_1(id):
    count_vector = np.zeros((1, 6)).ravel()
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    index = trans10to5(id, 5)

    flag2 = 1
    flag3 = 1

    for j in range(len(index)):
        count_vector[6 - len(index) + j] = index[j]

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = count_vector[5 - num]

    neigh2_cnts = con_mat @ con_mat
    neigh3_cnts = neigh2_cnts @ con_mat
    neigh4_cnts = neigh3_cnts @ con_mat
    connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

    for node in range(3):
        if connection_graph[node, 3] == 0:  # if any node doesnt send information to the last layer, remove it
            flag2 = 0
    # if flag2 == 0: return -1

    for node in range(3):
        if connection_graph[0, node + 1] == 0:  # if any node doesnt get information from the input layer, remove it
            flag3 = 0
        if flag3 == 0: return -1

    # 保证node1到node4一定有连接
    if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

    return con_mat

def get_cntmat_2(id):
    count_vector = np.zeros((1, 6)).ravel()
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    index = trans10to5(id, 5)

    flag2 = 1
    flag3 = 1

    for j in range(len(index)):
        count_vector[6 - len(index) + j] = index[j]

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = count_vector[5 - num]

    neigh2_cnts = con_mat @ con_mat
    neigh3_cnts = neigh2_cnts @ con_mat
    neigh4_cnts = neigh3_cnts @ con_mat
    connection_graph = con_mat + neigh2_cnts + neigh3_cnts + neigh4_cnts

    for node in range(3):
        if connection_graph[node, 3] == 0:  # if any node doesnt send information to the last layer, remove it
            flag2 = 0
    # if flag2 == 0: return -1

    for node in range(3):
        if connection_graph[0, node + 1] == 0:  # if any node doesnt get information from the input layer, remove it
            flag3 = 0
        # if flag3 == 0: return -1

    # 保证node1到node4一定有连接
    if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

    return con_mat

def get_cntmat_3(id):
    count_vector = np.zeros((1, 6)).ravel()
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    index = trans10to5(id, 5)

    flag2 = 1
    flag3 = 1

    for j in range(len(index)):
        count_vector[6 - len(index) + j] = index[j]

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = count_vector[5 - num]

    return con_mat

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0) #一个batch里算对了几个

        res.append(correct_k.mul_(100.0 / batch_size)) # 平均准确率用百分之几表示
    return res

path = '/home/sungengchen/NASSNN/snapshots_tinyimagenet100'
def save_checkpoint(arg, state, id, hist, model_param, time):
    if not os.path.exists(path + "/{}".format(id)):
        os.makedirs(path + "/{}".format(id))
    filename = os.path.join(path + "/{}/ckpt.pth".format(id))
    torch.save(state, filename)
    with open(path + "/{}/train_process.txt".format(id), 'w', encoding='utf-8') as f:
        f.write('architecture id: {}\r\n'.format(id))
        f.write('model parameters: {}\r\n'.format(model_param))
        f.write('epoch, train_acc, train_top5, train_loss, val_acc, val_top5, val_loss\r\n')
        for i in range(arg.epochs):
            for num, j in enumerate(hist[i]):
                if num != len(hist[i]) - 1:
                    f.write('{:.3f}, '.format(j))
                else:
                    f.write('{:.3f}\r\n'.format(j))

        f.write('test top1: {:.3f}, test tpo5: {:.3}\r\n'.format(hist[arg.epochs][0], hist[arg.epochs][1]))

        f.write('firing rate of layers: ')
        for num, k in enumerate(hist[arg.epochs + 1]):
            if num != len(hist[arg.epochs + 1]) - 1:
                f.write('{:.3f}, '.format(k))
            else:
                f.write('{:.3f}\r\n'.format(k))

        f.write('Elapsed time: {:.1f}hour{:.1f}minute{:.1f}second\r\n'.format(time[0], time[1], time[2]))


def data_transforms(args):
    if args.dataset == 'cifar10':
        MEAN = [0.4913, 0.4821, 0.4465]
        STD = [0.2470, 0.2434, 0.2615]
    elif args.dataset == 'cifar100':
        MEAN = [0.5071, 0.4867, 0.4408]
        STD = [0.2673, 0.2564, 0.2762]
    elif args.dataset == 'tinyimagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if (args.dataset == 'tinyimagenet'):
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:  # cifar10 or cifar100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    return train_transform, valid_transform


def random_choice(num_choice, layers):
    return list(np.random.randint(num_choice, size=layers))


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' % (hour, minute, second))
    return [hour, minute, second]


class Get_CIFAR10_train_val():
    base_folder = 'cifar-10-batches-py'
    data_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    def __init__(self, root, transform: Optional[callable] = None, target_transform: Optional[Callable] = None,
                 train=True):
        # super(Get_CIFAR10_train_val, self).__init__(root, transform=transform)
        super(Get_CIFAR10_train_val, self).__init__()
        self.root = root
        self.train_data = []
        self.val_data = []
        self.train_label = []
        self.val_label = []
        self.data = []
        self.label = []
        train_list = self.data_list[0:2]
        val_list = self.data_list[2:4]
        spilt_list = self.data_list[4]

        self.transform = transform
        self.target_transform = target_transform

        for file_name, checksum in train_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.train_data.append(entry['data'])
                self.train_label.extend(entry['labels'])

        for file_name, checksum in val_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.val_data.append(entry['data'])
                self.val_label.extend(entry['labels'])

        index_spilt_label = []
        file_name, checksm = spilt_list
        file_path = os.path.join(self.root, self.base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            spilt_data = entry['data']
            spilt_label = entry['labels']
            spilt_train_data = spilt_data[0:5000]
            spilt_train_label = spilt_label[0:5000]
            spilt_val_data = spilt_data[5000:]
            spilt_val_label = spilt_label[5000:]
        self.train_data.append(spilt_train_data)
        self.train_label += spilt_train_label
        self.val_data.append(spilt_val_data)
        self.val_label += spilt_val_label

        self.train_data = np.vstack(self.train_data).reshape(-1, 3, 32, 32)
        self.val_data = np.vstack(self.val_data).reshape(-1, 3, 32, 32)

        if train:
            self.data = self.train_data
            self.label = self.train_label
        else:
            self.data = self.val_data
            self.label = self.val_label
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

def get_module_paramerters(net):
    type_size = 4  # float占4个字节
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

