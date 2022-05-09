from itertools import cycle
import torch 
print(torch.__version__)
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import date
import cv2
import os
import os.path
import pickle
import torchvision
from torch.autograd import Variable
import random
from models.generator import Generator
from models.discriminator import Discriminator
from models.img_pool import ImgPool
import ignite.distributed as idist
from ignite.handlers import Checkpoint
from ignite.metrics import FID, InceptionScore, RunningAverage
import PIL.Image as Image
from ignite.contrib.handlers import ProgressBar
import logging
from ignite.engine import Engine, Events
import ignite
from pizzamaker.syntheticPizza import SyntheticPizzaDataset
import csv
from models.cut_nets import ResnetGenerator

ignite.utils.manual_seed(999)
ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)



device = idist.device()

"""G_A = Generator().to(device)
G_B = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)
"""
G_A = Generator().to(device)
G_B = Generator().to(device)
G = ResnetGenerator().to(device)
data_root = './data/pizza/pizza/'
image_size=(256,256)
bs=5

# currently test: cut b2a live



synprerec_test_ds = dset.ImageFolder(root=data_root+'syn_prerec_test/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
syn_test_dl = idist.auto_dataloader(synprerec_test_ds, batch_size=bs,drop_last=True, shuffle=True)

syn_live_test_ds = dset.ImageFolder(root='./data/pizza_Live_Test/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

syn_live_test_dl = idist.auto_dataloader(syn_live_test_ds, batch_size=bs, drop_last=True, shuffle=True)

syn_live_test_dl = idist.auto_dataloader(syn_live_test_ds, batch_size=bs, drop_last=True, shuffle=True)
real_test_ds = dset.ImageFolder(root=data_root+'real_test/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
                                
                                
                                
real_val_ds = dset.ImageFolder(root=data_root+'pizza_val/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))



real_eval_ds = torch.utils.data.ConcatDataset([real_test_ds, real_val_ds])
real_test_dl = idist.auto_dataloader(real_eval_ds, batch_size=bs,drop_last=True, shuffle=True)

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img.cpu())
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)
    
def evaluation_step_a2b(engine, batch):
    with torch.no_grad():
        a_real = batch[0][0].to(device)
        b_real = batch[1][0].to(device)
        G_A.eval()
        b_fake = G_A(a_real)

        if engine.state.iteration % 50 == 0:
            torchvision.utils.save_image(b_fake,f'./test_images/test_images_a2b_cycle/cycle_{engine.state.iteration}.png')

        fake = interpolate(b_fake)
        real = interpolate(b_real)
        return fake, real


def evaluation_step_b2a(engine, batch):
    with torch.no_grad():
        a_real = batch[0][0].to(device)
        b_real = batch[1][0].to(device)
        G_B.eval()
        a_fake = G_B(b_real)

        if engine.state.iteration % 50 == 0:
            torchvision.utils.save_image(a_fake,f'./test_images/test_images_b2a_cycle/cycle_{engine.state.iteration}.png')

        fake = interpolate(a_fake)
        real = interpolate(a_real)
        return fake, real

def evaluation_step_cut(engine, batch):
    with torch.no_grad():
        a_real = batch[0][0].to(device)
        b_real = batch[1][0].to(device)

        real = torch.cat((a_real, b_real), dim=0)


        G.eval()
        fake = G(real)
        b_fake = fake[:a_real.size(0)]
        idt_b = fake[a_real.size(0):]


        if engine.state.iteration % 50 == 0:
            torchvision.utils.save_image(b_fake,f'./test_images/test_images_a2b_cut/cut2_{engine.state.iteration}.png')

        fake = interpolate(b_fake)
        real = interpolate(b_real)
        return fake, real
FIDScores = []
ISScores = []

FID_Score = FID(device=idist.device())
IS_Score = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])


to_save_cycle = {"G_A": G_A, 'G_B': G_B}
to_save_cut = {"G": G}

to_load = to_save_cut


start =1
stop = 23

cut_path = 'cut_improved_fid_live_is.csv'
cycle_path = 'cycle_base_live_fid_is.csv'


with open(cut_path, 'a+',newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Epoch', 'FID', 'IS'])



for i in range(start, stop+1):


    #checkpoint_fp = f"./../../../../users/ram425/cut_models_improved/checkpoint_{i}.pt"
    checkpoint_fp = f"./cut_models_improved_live/checkpoint_{i}.pt"
    checkpoint = torch.load(checkpoint_fp)
    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)


    eval_obj_a2b = Engine(evaluation_step_a2b)
    FID_Score.attach(eval_obj_a2b, "fid")
    IS_Score.attach(eval_obj_a2b, "is")
    

    eval_obj_b2a = Engine(evaluation_step_b2a)
    FID_Score.attach(eval_obj_b2a, "fid")
    IS_Score.attach(eval_obj_b2a, "is")

    eval_obj_cut = Engine(evaluation_step_cut)
    FID_Score.attach(eval_obj_cut, "fid")
    IS_Score.attach(eval_obj_cut, "is")
    
    #ProgressBar().attach(eval_obj_a2b)
    #ProgressBar().attach(eval_obj_b2a)
    ProgressBar().attach(eval_obj_cut)
    @eval_obj_a2b.on(Events.EPOCH_COMPLETED)
    def display_score_a2b(engine):


        metrics = eval_obj_a2b.state.metrics
        fid_score = metrics['fid']
        is_score = metrics['is']
        FIDScores.append(fid_score)
        ISScores.append(is_score)

        

        print(f"Epoch [{engine.state.epoch}] Metric Scores for A->B")
        print(f"*   FID : {fid_score:4f}")
        print(f"*    IS : {is_score:4f}")
        with open(cycle_path, 'a+',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['AB',i, fid_score, is_score])

        
    @eval_obj_b2a.on(Events.EPOCH_COMPLETED)
    def display_score_b2a(engine):


        metrics = eval_obj_b2a.state.metrics
        fid_score = metrics['fid']
        is_score = metrics['is']
        FIDScores.append(fid_score)
        ISScores.append(is_score)

        

        print(f"Epoch [{engine.state.epoch}] Metric Scores for B->A")
        print(f"*   FID : {fid_score:4f}")
        print(f"*    IS : {is_score:4f}")
        with open(cycle_path, 'a+',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['BA',i, fid_score, is_score])

    @eval_obj_cut.on(Events.EPOCH_COMPLETED)
    def display_score_cut(engine):


        metrics = eval_obj_cut.state.metrics
        fid_score = metrics['fid']
        is_score = metrics['is']
        FIDScores.append(fid_score)
        ISScores.append(is_score)

        

        print(f"Epoch [{engine.state.epoch}] Metric Scores for A2B cut improved live")
        print(f"*   FID : {fid_score:4f}")
        print(f"*    IS : {is_score:4f}")
        with open(cut_path, 'a+',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['AB',i, fid_score, is_score])

    
    #eval_obj_a2b.run(zip(syn_live_test_dl, real_test_dl),max_epochs=1)
    #eval_obj_b2a.run(zip(syn_test_dl, real_test_dl),max_epochs=1)
    eval_obj_cut.run(zip(syn_live_test_dl,real_test_dl),max_epochs=1)



