import glob
import torch.nn as nn
import scipy.io as sio
import os
import torch.utils.data as data
import random
import torch
import numpy as np


class CAVE_dataset(torch.utils.data.Dataset):
    def __init__(self, args, train=True):
        super(CAVE_dataset, self).__init__()
        self.args = args
        self.train = train
        if train:
            self.scene_list = os.listdir(args.data_dir)[:20]
        else:
            self.scene_list = os.listdir(args.data_dir)[20:]

    def get_patch(self, HrHSI, HrLSI, LrHSI):
        ix = random.randrange(0, 13)
        iy = random.randrange(0, 13)
        iX = ix * 32
        iY = iy * 32

        HrHSI = np.ascontiguousarray(HrHSI[iX:iX+96, iY:iY+96, :])
        HrLSI = np.ascontiguousarray(HrLSI[iX:iX+96, iY:iY+96, :])
        LrHSI = np.ascontiguousarray(LrHSI[ix:ix+3,  iy:iy+3, :])

        # augumentation
        if random.random() < 0.5:
            HrHSI = np.ascontiguousarray(np.flip(HrHSI, [0]))
            HrLSI = np.ascontiguousarray(np.flip(HrLSI, [0]))
            LrHSI = np.ascontiguousarray(np.flip(LrHSI, [0]))

        if random.random() < 0.5:
            HrHSI = np.ascontiguousarray(np.flip(HrHSI, [1]))
            HrLSI = np.ascontiguousarray(np.flip(HrLSI, [1]))
            LrHSI = np.ascontiguousarray(np.flip(LrHSI, [1]))

        if random.random() < 0.5:
            HrHSI = np.ascontiguousarray(HrHSI.transpose(1, 0, 2))
            HrLSI = np.ascontiguousarray(HrLSI.transpose(1, 0, 2))
            LrHSI = np.ascontiguousarray(LrHSI.transpose(1, 0, 2))

        return HrHSI, HrLSI, LrHSI

    def __getitem__(self, index):
        if self.train:
            index = index % 20
        data = sio.loadmat(os.path.join(self.args.data_dir, self.scene_list[index])+'/'+self.scene_list[index]+'.mat')
        HrHSI = data['HrHSI']
        HrLSI = data['HrLSI']
        LrHSI = data['LrHSI']

        if self.train:
            HrHSI, HrLSI, LrHSI = self.get_patch(HrHSI, HrLSI, LrHSI)

        HrHSI = torch.from_numpy(HrHSI.transpose(2, 0, 1).astype(np.float)).float()
        HrLSI = torch.from_numpy(HrLSI.transpose(2, 0, 1)).float()
        LrHSI = torch.from_numpy(LrHSI.transpose(2, 0, 1)).float()

        HrHSI = (HrHSI) / (2 ** 16 - 1)
        HrLSI = (HrLSI) / (2 ** 16 - 1)
        LrHSI = (LrHSI) / (2 ** 16 - 1)

        return HrHSI, HrLSI, LrHSI

    def __len__(self):
        if self.train:
            return self.args.n_iters * self.args.batch_size
        if not self.train:
            return len(self.scene_list)
