import os
import pickle
import time
from typing import Optional, Callable, Tuple, Any

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


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
    # if con_mat[0, 3] == 0: return -1  # ensure direct connection between input=>output for fast information propagation

    return con_mat

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(arg, state, id, hist, model_param, time):
    if not os.path.exists("./snapshots/{}".format(id)):
        os.makedirs("./snapshots/{}".format(id))
    filename = os.path.join("./snapshots/{}/ckpt.pth".format(id))
    torch.save(state, filename)
    with open("./snapshots/{}/train_process.txt".format(id), 'w', encoding='utf-8') as f:
        f.write('architecture id: {}\r\n'.format(id))
        f.write('model parameters: {}\r\n'.format(model_param))
        f.write('epoch, train_acc, train_loss, val_acc, val_loss\r\n')
        for i in range(arg.epochs):
            for num, j in enumerate(hist[i]):
                if num != len(hist[i]) - 1:
                    f.write('{:.3f}, '.format(j))
                else:
                    f.write('{:.3f}\r\n'.format(j))

        f.write('test accuracy: {:.3f}\r\n'.format(hist[arg.epochs]))

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

def anyToDecimal(num, n):
    baseStr = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "a": 10, "b": 11, "c": 12, "d": 13, "e": 14, "f": 15, "g": 16, "h": 17, "i": 18, "j": 19}
    new_num = 0
    nNum = len(num) - 1
    for i in num:
        new_num = new_num + baseStr[i] * pow(n, nNum)
        nNum = nNum - 1
    return new_num

def encoding_to_id(encoding):
    encoding_re = encoding[::-1]
    id = anyToDecimal(''.join('%s' %a for a in encoding_re), 5)

    return id

def encoding_to_cntmat(encoding):
    con_mat = np.zeros((4, 4))
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for num, (k0, k1) in enumerate(position):
        con_mat[k0, k1] = encoding[num]

    return con_mat







