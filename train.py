import os
import sys
import time
import torch
import torch.nn
from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR
import argparse
# config
parser = argparse.ArgumentParser()
parser.add_argument('--w1', type=int, default=10)
parser.add_argument('--w2', type=int, default=10)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--compress', type=str, default='c40')  # ['raw','c23','c40]
parser.add_argument('--bz', type=int, default=64)
parser.add_argument('--pretrained_path', type=str, default='../outputs/xception-b5690688.pth')
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--mode', type=str, default='Original')  # ['Original', 'Two_Stream','FAM','Decoder','CLloss']
parser.add_argument('--model_path', type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
compress = args.compress
dataset_path = '/mnt/sda/FF_plus_plus/result1_{}'.format(compress)
pretrained_path = args.pretrained_path
batch_size = args.bz
gpu_ids = [*range(osenvs)]
max_epoch = args.epoch
loss_freq = 300
mode = args.mode
ckpt_dir = '../FACL/log'
ckpt_name = "{}_{}_bz{}".format(mode, max_epoch, batch_size)


if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, "youtube", compress, "train"), size=299, frame_num=300)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=16)
    
    len_dataloader = dataloader_real.__len__()
    print("len_dataloader={}".format(len_dataloader))

    dataset_img, total_len = get_dataset(name='train', size=299, root=dataset_path, frame_num=300, compress=compress)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=16
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, args.w1, args.w2, args.lr)
    if args.model_path:
        model.load(model_path)
    model.total_steps = 0
    epoch = 0
    best_auc = 0
    epoch_loss = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=6)

    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)
        
        logger.debug(f'No {epoch} last loss: {epoch_loss / len_dataloader}')
        i = 0
        epoch_loss = 0
        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1

            try:
                data_real = real_iter.next()
                data_fake = fake_iter.next()
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real[0].shape[0] != data_fake[0].shape[0]:
                continue

            bz = data_real[0].shape[0]
            
            data = torch.cat([data_real[0],data_fake[0]],dim=0)
            mask = torch.cat([data_real[1],data_fake[1]],dim=0)
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0),torch.ones(bz).unsqueeze(dim=0)],dim=1).squeeze(dim=0)

            # manually shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            mask = mask[idx]
            label = label[idx]

            data = data.detach()
            mask = mask.detach()
            label = label.detach()

            model.set_input(data, label, mask)
            loss = model.optimize_weight()
            epoch_loss += loss.item()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')

            if i % int(len_dataloader / 2) == 0:
                model.model.eval()
                auc, r_acc, f_acc, acc = evaluate(model, dataset_path, mode='val', compress=compress)
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, acc:{(r_acc + 4 * f_acc) / 5}, acc:{acc}')
                if auc > best_auc:
                    best_auc = auc
                    pp = "outputs/mode-{}_w1-{}_w2-{}_compress-{}".format(args.mode, args.w1, args.w2, compress)
                    if not os.path.exists(pp):
                        os.mkdir(pp)
                    model.save("{}/{}_{}_{:5f}.pth".format(pp, epoch, model.total_steps, auc))
                    print("Model Saved Successfully!")
                model.model.train()
        epoch = epoch + 1
        scheduler.step()

    model.model.eval()
    auc, r_acc, f_acc, acc = evaluate(model, dataset_path, mode='test', compress=compress)
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, acc:{(r_acc + 4 * f_acc) / 5}, acc:{acc}')
