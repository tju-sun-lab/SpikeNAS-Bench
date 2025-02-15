import numpy as np
import os
import random
import re
from utils import get_cntmat_3


def get_train_log(id: int, snapshots_dir: str):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    firing_rate_layers = []
    file_dir = snapshots_dir + '/' + str(id)
    txt_dir = file_dir + '/train_process.txt'
    model_dir = file_dir + '/ckpt.pth'
    fo = open(txt_dir, 'r')
    num_lines = len(fo.readlines())
    if num_lines > 200:
        red_line = 203
    else:
        red_line = 103
    fo.close()
    fo = open(txt_dir, 'r')
    for num, line in enumerate(fo.readlines()):
        if num == 0:
            line = line.strip('\n')
            line_spilt = line.split(' ')
            architecture_id = int(line_spilt[2])
            cnt_mat = get_cntmat_3(architecture_id)
        elif num == 1:
            line = line.strip('\n')
            line_spilt = line.split(' ')
            model_param = int(line_spilt[2])
        elif num == 2:
            continue
        elif num == red_line:
            line = line.strip('\n')
            line_spilt = line.split(' ')
            test_acc = float(line_spilt[2])
        elif num == red_line + 1:  # 204/104
            line = line.strip('\n')
            line_spilt = line.split(' ')
            for j in range(4, 20):
                if len(line_spilt[j]) == 5:
                    firing_rate_layers.append(float(line_spilt[j]))
                elif len(line_spilt[j]) == 6:
                    firing_rate_layers.append(float(line_spilt[j][0:5]))
                elif len(line_spilt[j]) == 7:
                    firing_rate_layers.append(float(line_spilt[j][0:6]))
        elif num == red_line + 2:  # 205/105
            time = re.findall(r"\d+\.?\d*", line)
            time_s = float(time[0]) * 3600 + float(time[1]) * 60 + float(time[2])
        else:
            line_spilt = re.findall(r"\d+\.?\d*", line)
            train_acc.append(float(line_spilt[1]))
            train_loss.append(float(line_spilt[2]))
            val_acc.append(float(line_spilt[3]))
            val_loss.append(float(line_spilt[4]))

    return architecture_id, cnt_mat, model_param, train_acc, train_loss, val_acc, val_loss, test_acc, \
           firing_rate_layers, time_s


if __name__ == '__main__':
    architecture_id, cnt_mat, model_param, train_acc, train_loss, val_acc, val_loss, test_acc, firing_rate_layers, time_s \
        = get_train_log(3276, 'snapshots')
