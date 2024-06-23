import torch
from option import args
from data.dataset import CAVE_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model import HyperSR
import numpy as np
import os
from sewar.full_ref import psnr
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def save_ckpt(state, save_path='./log', filename='checkpoint.pth'):
    torch.save(state, os.path.join(save_path, filename))


def train(train_loader, args):
    net = HyperSR(channels_LSI=3, channels_HSI=31, channels=64, n_endmembers=64).cuda()
    net = nn.DataParallel(net, device_ids=list(range(args.n_GPUs)))
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    loss_list = []
    psnr_list = []
    loss_save = []
    for idx_iter, [HrHSI, HrLSI, LrHSI] in enumerate(train_loader):
        HrHSI = HrHSI.cuda()
        HrLSI = HrLSI.cuda()
        LrHSI = LrHSI.cuda()

        # inference
        rec_HrHSI = net(HrLSI, LrHSI)

        # losses
        loss = nn.L1Loss()(rec_HrHSI, HrHSI)
        loss_list.append(loss.data.cpu())
        psnr_list.append(psnr(HrHSI.data.cpu().numpy(), rec_HrHSI.data.cpu().numpy(), MAX=1))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print
        if idx_iter % 200 == 0:
            print('iteration %5d of total %5d, loss---%f, psnr---%f' %
                  (idx_iter + 1, args.n_iters, float(np.array(loss_list).mean()), float(np.array(psnr_list).mean())))
            loss_save.append(np.array(loss_list).mean())

        if idx_iter % 1000 == 0:
            save_ckpt({
                'iter': idx_iter + 1,
                'state_dict': net.state_dict(),
                'loss': loss_save,
            }, save_path='log/', filename=args.dataset+'/'+args.model+'_'+str(args.n_endmembers)+'_iter' + str(idx_iter + 1) + '.pth')
            loss_list = []
            psnr_list = []

        scheduler.step()


if __name__ == '__main__':
    # dataloader
    train_set = CAVE_dataset(args, train=True)
    train_loader = DataLoaderX(dataset=train_set, num_workers=6, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # train
    train(train_loader, args)

