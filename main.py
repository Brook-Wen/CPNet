import os
import torch
import random
import argparse
import numpy as np
from solver import Solver
from dataset.dataset import get_loader


def get_test_info(sal_mode='360sod'):
    if sal_mode == '360sod':
        image_root = './data/test/360SOD/'
        image_source = './data/test/360SOD/test.lst'
    elif sal_mode == '360ssod':
        image_root = './data/test/360SSOD/'
        image_source = './data/test/360SSOD/test.lst'
    elif sal_mode == '360isod':
        image_root = './data/test/F-360iSOD/'
        image_source = './data/test/F-360iSOD/test.lst'

    return image_root, image_source


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    resnet_path = './pretrained/resnet50.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--betas', type=float, default=[0.9, 0.999])
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=76)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=2)
    parser.add_argument('--iter_size', type=int, default=8)
    parser.add_argument('--show_every', type=int, default=50)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cpu_num_thread', type=str, default='8')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    # Train data
    parser.add_argument('--train_root', type=str, default='./data/train')
    parser.add_argument('--train_list', type=str, default='./data/train/train.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--test_fold', type=str, default=None)
    parser.add_argument('--sal_mode', type=str, default='360sod')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    os.environ["OMP_NUM_THREADS"] = config.cpu_num_thread

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list


    main(config)
