import argparse
import os
import io
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import PIL.Image as pil_image
from model import REDNet10, REDNet20, REDNet30
from dataset import Dataset
from utils import AverageMeter
from tqdm import tqdm
import numpy as np

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=10)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    cool = opt.outputs_dir + "/cool"
    if not os.path.exists(cool):
        os.makedirs(cool)


    val_dir = os.fsencode(opt.images_dir)

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    val_set = Dataset(opt.images_dir, 50, opt.jpeg_quality, True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)

    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss()

    # filename = os.path.basename(opt.image_path).split('.')[0]
    
    smallest = [float('inf'), None, None]
    largest = [0, None, None]

    for file in os.listdir(val_dir):
        filename = os.fsdecode(file)
        input = pil_image.open(os.path.join(opt.images_dir, filename)).convert('RGB')
        orig = input.copy()
        # input.save(os.path.join(opt.outputs_dir, '{}_jpeg_orig.png'.format(filename)))

        # buffer = io.BytesIO()
        # input.save(buffer, format='jpeg', quality=opt.jpeg_quality)
        # input = pil_image.open(buffer)

        new_dim = (160, 128)
        w, h = input.size
        input = input.resize(new_dim, resample = pil_image.NEAREST)
        input = input.resize((w, h), resample = pil_image.NEAREST)
        # input.save(os.path.join(opt.outputs_dir, '{}_jpeg_small.png'.format(filename)))

        
        input_tens = transforms.ToTensor()(input).unsqueeze(0).to(device)
        orig_tens = transforms.ToTensor()(orig).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tens)

        loss = criterion(pred, orig_tens)

        pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        output = pil_image.fromarray(pred, mode='RGB')
        # output.save(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, opt.arch)))

        loss_curr = loss.item()
        acc = 100.0 - loss_curr

        img_size = orig.size

        new_image = pil_image.new('RGB',(3*img_size[0], img_size[1]), (250,250,250))
        new_image.paste(input,(0,0))
        new_image.paste(output,(img_size[0],0))
        new_image.paste(orig,(2*img_size[0],0))
        new_image.save(os.path.join(opt.outputs_dir, '{}_results_acc_{}.png'.format(filename, str(acc))))

        if acc < smallest[0]:
            smallest = [acc, new_image, filename]
        if acc > largest[0]:
            largest = [acc, new_image, filename]

largest_img = largest[1]
largest.save(os.path.join(cool, '{}_largest_acc_{}.png'.format(largest[2], str(largest[0]))))

smallest_img = smallest[1]
smallest.save(os.path.join(cool, '{}_smallest_acc_{}.png'.format(smallest[2], str(smallest[0]))))