import argparse
import os
import random
import time
import datetime
import warnings

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.diffusion import DenoiseDiffusion
from models.unet import UNet

parser = argparse.ArgumentParser(description='PyTorch DDPM Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1_000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--steps', default=1_000, type=int, metavar='N',
                    help='number of time step')
parser.add_argument('--samples', default=16, type=int, metavar='N',
                    help='number of samples to generate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    summary = SummaryWriter()

    # Options
    image_channels = 1
    image_size = 32
    n_channels = 64
    channels_multipliers = [1, 2, 2, 4]
    is_attention = [False, False, True, True]

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Models
    eps_model = UNet(img_channels=image_channels,
                     n_channels=n_channels,
                     ch_mults=channels_multipliers,
                     is_attn=is_attention).cuda(args.gpu)
    diffusion = DenoiseDiffusion(eps_model=eps_model,
                                 n_steps=args.steps,
                                 args=args).cuda(args.gpu)

    # Optimizer / criterion(wSDR)
    criterion = nn.MSELoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=args.lr)

    # Dataset / Dataloader
    train_dataset = datasets.MNIST(
        root="./dataset", train=True, download=True,
        transform=transforms.Compose([transforms.Resize(image_size),
                                      transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Train
    param = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print("Total Param: ", param)

    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader=train_loader,
              diffusion=diffusion,
              criterion=criterion,
              optimizer=optimizer,
              summary=summary,
              epoch=epoch,
              args=args)

        # Save Sample
        sample(epoch=epoch,
               diffusion=diffusion,
               image_channels=image_channels,
               image_size=image_size,
               args=args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            torch.save({
                'epoch': epoch + 1,
                'diffusion': diffusion.state_dict(),
                'optimizer': optimizer.state_dict()
            }, "saved_models/checkpoint_%d.pth" % (epoch + 1))


def train(train_loader, diffusion, criterion, optimizer, summary, epoch, args):
    end = time.time()
    diffusion.train()
    for i, (image, _) in enumerate(train_loader):
        image = image.cuda(args.gpu, non_blocking=True)

        noise, eps_theta = diffusion(image)
        loss = criterion(noise, eps_theta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        niter = epoch * len(train_loader) + i
        summary.add_scalar('Train/loss', loss.item(), niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | Loss: %f |"
                  % (epoch + 1, i, len(train_loader), loss))

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


def sample(epoch, diffusion, image_channels, image_size, args):
    diffusion.eval()

    with torch.no_grad():
        x = torch.randn([args.samples, image_channels, image_size, image_size]).cuda(args.gpu)

        for t_ in range(args.steps):
            t = args.steps - t_ - 1
            x = diffusion.p_sample(x, x.new_full((args.samples, ), t, dtype=torch.long))

        torchvision.utils.save_image(x, "outputs/%03d.png" % (epoch), normalize=False)
        print("Save Image")


if __name__ == "__main__":
    main()


