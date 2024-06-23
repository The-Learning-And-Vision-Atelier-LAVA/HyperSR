import torch
from option import args
from data.dataset import CAVE_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model import HyperSR
from utility import metrics
import os


def test(test_loader, args):
    net = HyperSR(channels_LSI=3, channels_HSI=31, channels=64, n_endmembers=args.n_endmembers).cuda()
    net = nn.DataParallel(net, device_ids=list(range(args.n_GPUs)))
    net.eval()

    ckpt = torch.load('./log/' + args.dataset + '/' + args.model + '_' + str(args.n_endmembers) + '_iter30001.pth')
    net.load_state_dict(ckpt['state_dict'])

    psnr_sum = 0
    sam_sum = 0
    ergas_sum = 0
    ssim_sum = 0
    uqi_sum = 0

    for idx_iter, data in enumerate(test_loader):
        if args.dataset == 'CAVE':
            [HrHSI, HrLSI, LrHSI] = data
        else:
            [HrHSI, HrLSI, LrHSI, V] = data

        HrHSI = HrHSI.to(args.device)
        HrLSI = HrLSI.to(args.device)
        LrHSI = LrHSI.to(args.device)

        # inference
        rec_HrHSI = net(HrLSI, LrHSI)

        # CAVE
        if args.dataset == 'CAVE':
            data_range = 2 ** 16 - 1
            rec_HrHSI = (rec_HrHSI.clamp(0, 1) * data_range).round().div(data_range)

        psnr, sam, ergas, ssim, uqi = metrics(HrHSI.squeeze().data.cpu().numpy(), rec_HrHSI.squeeze().data.cpu().numpy(), 32)
        psnr_sum += psnr
        sam_sum += sam
        ergas_sum += ergas
        ssim_sum += ssim
        uqi_sum += uqi

if __name__ == '__main__':
    # dataloader
    test_set = CAVE_dataset(args, train=False)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False, drop_last=True, pin_memory=True)

    # test
    with torch.no_grad():
        test(test_loader, args)