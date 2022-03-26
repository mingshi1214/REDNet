import argparse
import os
from tokenize import PlainToken
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import REDNet10, REDNet20, REDNet30
from dataset import Dataset
from utils import AverageMeter
import matplotlib.pyplot as plt 
import numpy as np

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    parser.add_argument('--val_images_dir', type=str, required=True)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.patch_size, opt.jpeg_quality, opt.use_fast_loader)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    val_set = Dataset(opt.val_images_dir, opt.patch_size, opt.jpeg_quality, opt.use_fast_loader)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    loss_arr = []
    val_arr = []

    loss_data = 0
    data_cnt = 0
    val_loss_data = 0
    val_data_cnt = 0

    min_valid_loss = np.inf

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()
        val_losses = AverageMeter()
        loss_data = 0
        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                loss_data += epoch_losses.avg
                data_cnt += 1
                _tqdm.update(len(inputs))

            valid_loss = 0.0
            model.eval()     # Optional when not using Model Specific layer
            for val_data in val_loader:
                val_inputs, val_labels = data
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device) 

                preds = model(val_inputs)
                loss = criterion(preds, val_labels)
                val_losses.update(loss.item(), len(inputs))

                val_loss_data += val_losses.avg
                val_data_cnt += 1

        loss_epoch = loss_data/data_cnt
        loss_arr.append(loss_epoch)
        print("train loss: ", loss_epoch)
        
        val_loss_epoch = val_loss_data/val_data_cnt
        val_arr.append(val_loss_epoch)
        print("val loss: ", val_loss_epoch)

        if min_valid_loss > valid_loss:
            torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_weights.pth'.format(opt.arch)))

        if epoch == opt.num_epochs - 1:
            loss_accum = np.array(loss_arr)
            val_loss_accum = np.array(loss_arr)
            epochs = np.arange(0, epoch + 1)

            fig, ax = plt.subplots()

            line1 = ax.plot(epochs, loss_accum, label="Train")
            line2 = ax.plot(epochs, val_loss_accum, label="Validation")
            leg = ax.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Over Epochs")
            plt.savefig(os.path.join(opt.outputs_dir, 'loss.png'))