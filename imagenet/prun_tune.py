import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from datetime import datetime

import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import util

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")


# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')                               
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay', '--learning-rate-decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--lr_int', '--learning-rate-interval', default=30, type=int,
                    metavar='LRI', help='learning rate decay interval')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='saves', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='model', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--sensitivity', type=float, default=1e-4,
                    help="threshold used for pruning")

global args, best_prec1, save_path
args = parser.parse_args()
best_prec1 = 0

save_path = os.path.join(args.resume, 'finetune', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    raise OSError('Directory {%s} exists. Use a new one.' % save_path)

logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
logger = logging.getLogger('main')
logger.addHandler(logging.StreamHandler())
logger.info("Saving to %s", save_path)
logger.info("Running arguments: %s", args)

def main():
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

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            if args.arch.startswith('alexnetv2') and args.pretrained:
                pass
            else:    
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    input_size = 224

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    if args.pretrained is not True:
        model.load_state_dict(torch.load(os.path.join(args.resume, args.model+'.pth')))
        print('model loaded')

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)


        
    device = torch.device("cuda")    
    # Initial pruning
    print("--- Pruning ---")
    masks = {}
    for name, p in model.named_parameters():
        if 'weight' in name:
            tensor = p.data.cpu().numpy()
            threshold = args.sensitivity 
            new_mask = np.where(abs(tensor) < threshold, 0, tensor)
            mask = np.where(abs(tensor) < threshold, 0., 1.)
            masks[name] = torch.from_numpy(mask).float().to(device)
            p.data = torch.from_numpy(new_mask).to(device)        

    util.print_nonzeros(model)
    print('Pruned model evaluation...')
    prec1, prec5 = validate(val_loader, model, criterion)
    
    if args.evaluate:
        return
    
    best_prec1 = prec5
    torch.save(model.state_dict(), os.path.join(save_path, 'finetuned.pth'))

    print("--- Finetuning ---")
    
    curves = np.zeros(((args.epochs-args.start_epoch)*(len(train_loader)//args.print_freq),2))
    valid = np.zeros(((args.epochs-args.start_epoch),3))
    step = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        curves,step = train(train_loader, model, criterion, optimizer, epoch, curves, step, masks)

        # evaluate on validation set
        valid[epoch, 0] = epoch
        valid[epoch, 1], valid[epoch, 2] = validate(val_loader, model, criterion)
        prec5 = valid[epoch, 2]
        
        # plot training curve
        np.savetxt(os.path.join(save_path, 'curves.dat'), curves)
        
        clr1 = (0.5, 0., 0.)
        clr2 = (0.0, 0.5, 0.)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('steps')
        ax1.set_ylabel('Loss', color=clr1)
        ax1.tick_params(axis='y', colors=clr1)
        
        start = 0
        end = step
        markersize = 12
        coef = 2.
        ax1.plot(curves[start:end, 0], curves[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
        
        #ax2.set_ylim(bottom=20, top=100)
        ax1.legend(('Train loss'), loc='lower right')
        fig.savefig(os.path.join(save_path, 'loss-vs-steps.pdf'))
        
        # plot validation curve
        np.savetxt(os.path.join(save_path, 'valid.dat'), valid)
        
        fig3, ax5 = plt.subplots()
        ax6 = ax5.twinx()
        ax5.set_xlabel('epochs')
        ax5.set_ylabel('Acc@1', color=clr1)
        ax5.tick_params(axis='y', colors=clr1)
        ax6.set_ylabel('Acc@5', color=clr2)
        ax6.tick_params(axis='y', colors=clr2)
        
        start = 0
        end = epoch+1
        markersize = 12
        coef = 2.
        ax5.plot(valid[start:end, 0], valid[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
        ax6.plot(valid[start:end, 0], valid[start:end, 2], '-', color=[c*coef for c in clr2], markersize=markersize)
        
        #ax2.set_ylim(bottom=20, top=100)
        ax5.legend(('Acc@1'), loc='lower right')
        ax6.legend(('Acc@5'), loc='lower left')
        fig3.savefig(os.path.join(save_path, 'accuracy-vs-epochs.pdf'))
        
        if prec5 > best_prec1:
            torch.save(model.state_dict(), os.path.join(save_path, 'finetuned.pth'))
            best_prec1 = prec5
            print('New best performance')
            
    print("--- Evaluating ---")        
    model.load_state_dict(torch.load(os.path.join(save_path, 'finetuned.pth')))
    prec1 = validate(val_loader, model, criterion)


def train(train_loader, model, criterion, optimizer, epoch, curves, step, mask):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        device = torch.device("cuda") 
        for name, p in model.named_parameters():
            if 'weight' in name:
                p.grad.data = p.grad.data*mask[name]
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #break
        if i and i % args.print_freq == 0:
            curves[step, 0] = len(train_loader)*epoch+i
            curves[step, 1] = losses.avg
            
            step += 1
                   
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return curves, step



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))


        nonzero = total = 0
        filter_count = filter_total = 0
        total_sparsity = total_layer = 0
        for name, p in model.named_parameters():
            if 'weight' in name and len(list(p.size()))>1:
                tensor = p.data.cpu().numpy()
                tensor = np.abs(tensor)
                nz_count = np.count_nonzero(tensor)
                total_params = np.prod(tensor.shape)
                nonzero += nz_count
                total += total_params
                
                if len(tensor.shape)==4:
                    dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
                    dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
                    nz_count0 = np.count_nonzero(dim0)
                    nz_count1 = np.count_nonzero(dim1)
                    filter_count += nz_count0*nz_count1
                    filter_total += len(dim0)*len(dim1)
                    total_sparsity += 1-(nz_count0*nz_count1)/(len(dim0)*len(dim1))
                    total_layer += 1
                if len(tensor.shape)==2:
                    dim0 = np.sum(tensor, axis=0)
                    dim1 = np.sum(tensor, axis=1)
                    nz_count0 = np.count_nonzero(dim0)
                    nz_count1 = np.count_nonzero(dim1)
                    filter_count += nz_count0*nz_count1
                    filter_total += len(dim0)*len(dim1)
                    total_sparsity += 1-(nz_count0*nz_count1)/(len(dim0)*len(dim1))
                    total_layer += 1
                
        elt_sparsity = (total-nonzero)/total
        input_sparsity = (filter_total-filter_count)/filter_total
        output_sparsity = total_sparsity/total_layer

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Sparsity elt: {elt_sparsity:.3f} str: {input_sparsity:.3f} str_avg: {output_sparsity:.3f}'
              .format(top1=top1, top5=top5, elt_sparsity=elt_sparsity, input_sparsity=input_sparsity, output_sparsity=output_sparsity))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_int))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
