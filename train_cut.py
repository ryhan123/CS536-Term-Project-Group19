
from itertools import filterfalse
from tkinter import FALSE
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
#from torchinfo import summary
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
from models.cut_nets import Discriminator, PatchNCELoss, PatchSampleF, ResnetGenerator
from models.img_pool import ImgPool
import ignite
from ignite.handlers import Checkpoint, DiskSaver
import PIL.Image as Image
import logging
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from pizzamaker.syntheticPizza import SyntheticPizzaDataset
import torch.nn.init as init
from ignite.metrics import RunningAverage

#device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#print(device)
data_root = './data/pizza/pizza/'
bs = 5
workers = 2
image_size = (256,256)
device = torch.device('cuda')
print(device)


def get_syn_live_data():


    """syn_live_train_ds = SyntheticPizzaDataset(transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))"""

    syn_live_train_ds = dset.ImageFolder(root='./data/pizzamaker_Dataset2/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    syn_live_train_dl = torch.utils.data.DataLoader(syn_live_train_ds, batch_size=bs, drop_last=True, shuffle=True)


    """syn_live_test_ds = SyntheticPizzaDataset(transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    syn_live_test_dl = idist.auto_dataloader(syn_live_test_ds, batch_size=bs, drop_last=True, shuffle=True)
    """

    return syn_live_train_dl



def get_syn_prerec_data():

    synprerec_train_ds = dset.ImageFolder(root=data_root+'syn_prerec_train/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    synprerec_train_dl = torch.utils.data.DataLoader(synprerec_train_ds, batch_size=bs, drop_last=True, shuffle=True)

    synprerec_test_ds = dset.ImageFolder(root=data_root+'syn_prerec_test/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    syn_val_ds = dset.ImageFolder(root=data_root+'syn_val/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))


    syn_eval_ds = torch.utils.data.ConcatDataset([synprerec_test_ds, syn_val_ds])


    syn_test_dl = torch.utils.data.DataLoader(syn_eval_ds, batch_size=bs,drop_last=True)


    return synprerec_train_dl, syn_test_dl



def get_real_data():
    real_train_ds = dset.ImageFolder(root=data_root+'real_train/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    real_train_dl = torch.utils.data.DataLoader(real_train_ds, batch_size=bs,drop_last=True, shuffle=True)

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
    real_test_dl = torch.utils.data.DataLoader(real_eval_ds, batch_size=bs,drop_last=True)

    return real_train_dl, real_test_dl



a_train_dl, a_test_dl = get_syn_prerec_data()
b_train_dl, b_test_dl = get_real_data()

G = ResnetGenerator().to(device)# translates images a to b
D= Discriminator().to(device) # determines whether an image is in a or not
F = PatchSampleF().to(device)
lr = 2e-4
beta1 = 0.5

lam_gan = 1
img_pool_a = ImgPool(size=50)
img_pool_b = ImgPool(size=50)

g_losses = []
d_losses = []
G_opt = optim.Adam(G.parameters(),lr=lr,betas=(beta1,0.999))
D_opt = optim.Adam(D.parameters(),lr=lr,betas=(beta1,0.999))


flip = True
flipped = False
num_patches = 256
lam_nce = 1
nce_idt = False
load_model = True




#print(summary(G_A, input_size=(bs, 3, 256, 256)))
GAN_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

nce_criterion = []

nce_layers = '0,4,8,12,16'
nce_layers = [int(i) for i in nce_layers.split(',')]
def D_Loss(b_real, b_fake):

    fake_prediction = D(b_fake.detach())
    d_fake_loss = generator_loss(fake_prediction, False) # add mean() back
    real_prediction = D(b_real)
    d_real_loss = generator_loss(real_prediction, True)
    d_loss = (d_fake_loss + d_real_loss) * 0.5

    return d_loss
for i in nce_layers:

    nce_criterion.append(PatchNCELoss(base=False).to(device))
"""
    calc D(A)

"""

def discriminator_loss(real, fake): 
    
    l_real = GAN_loss(real,torch.ones_like(real).float())
    l_gen = GAN_loss(fake, torch.zeros_like(fake).float())

    return 0.5*(l_real + l_gen)



def generator_loss(pred, is_real):
    
    """
        Generator Loss:
            - BCEWithLogitsLoss
    """
    target = torch.tensor(int(is_real), device=device).expand_as(pred).float()
    return GAN_loss(pred, target)

def cycle_consistency_loss(real, cycled):
    
    return l1_loss(cycled, real)

def identity_loss(real, copy):
    return l1_loss(copy, real)


lam_neighbors = 0.1
def nce_loss_improved(source, target):

    n_layers = len(nce_layers)

    feat_q = G(target, nce_layers, encode_only=True)

    if flip and flipped:
        feat_q = [torch.flip(f_q, [3]) for f_q in feat_q]

    feat_k = G(source, nce_layers, encode_only=True)


    feat_k_pool, sample_ids = F(feat_k, num_patches, None)

    #print(sample_ids)
    feat_q_pool, _ = F(feat_q,num_patches, sample_ids)
    total_nce_loss = 0.0

    total_neighbor_loss = 0.0
    #print('feat_k_pool:',feat_k_pool)
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, nce_criterion, nce_layers):
        loss_nce, loss_neighbors = crit(f_q, f_k)
        loss_nce *= lam_nce
        loss_neighbors *= lam_neighbors
        total_nce_loss += loss_nce.mean()
        total_neighbor_loss += loss_neighbors
        
    return total_nce_loss/n_layers , total_neighbor_loss/n_layers


def nce_loss_base(source, target):

    n_layers = len(nce_layers)

    feat_q = G(target, nce_layers, encode_only=True)

    if flip and flipped:
        feat_q = [torch.flip(f_q, [3]) for f_q in feat_q]

    feat_k = G(source, nce_layers, encode_only=True)


    feat_k_pool, sample_ids = F(feat_k, num_patches, None)

    #print(sample_ids)
    feat_q_pool, _ = F(feat_q,num_patches, sample_ids)
    total_nce_loss = 0.0

    total_neighbor_loss = 0.0
    #print('feat_k_pool:',feat_k_pool) #f_q is 1280, 256, fq_pool is a list of these
    for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, nce_criterion, nce_layers):
        loss_nce = crit(f_q, f_k)
        loss_nce *= lam_nce
        total_nce_loss += loss_nce.mean()
        
    return total_nce_loss/n_layers
"""
    Goal:
    

        - We want to encode our input and output
        - select random patches on each image in input batch
        - find the same patches on the output image
        - minimize NLL between input and output
        - 

"""


def G_Loss_improved(a_real, b_real, b_idt, b_fake):
    fake = b_fake
    if lam_gan > 0.0:
        fake_prediction = D(fake)

        loss_g_gan = generator_loss(fake_prediction, True).mean()*lam_gan
    else:
        loss_g_gan = 0

    loss_nce, loss_neighbors = nce_loss_improved(a_real, b_fake)

    loss_nce_y, loss_neighbors_y = nce_loss_improved(b_real, b_idt)
    loss_nce_both = (loss_nce_y + loss_nce)*0.5
    loss_neighbors_both = (loss_neighbors + loss_neighbors_y)*0.1
    
    loss_g = loss_g_gan + loss_nce_both + loss_neighbors_both

    return loss_g

def G_Loss_base_idt(a_real, b_real, b_idt, b_fake):
    fake = b_fake
    if lam_gan > 0.0:
        fake_prediction = D(fake)

        loss_g_gan = generator_loss(fake_prediction, True).mean()*lam_gan
    else:
        loss_g_gan = 0

    loss_nce = nce_loss_base(a_real, b_fake)

    loss_nce_y  = nce_loss_base(b_real, b_idt)
    loss_nce_both= (loss_nce_y + loss_nce)*0.5
    
    loss_g = loss_g_gan + loss_nce_both

    return loss_g


    # initialize F weights

init_batch_a = next(iter(a_train_dl))[0].to(device)
init_batch_b = next(iter(b_train_dl))[0].to(device)

real = torch.cat((init_batch_a, init_batch_b), dim=0)

if flip:
    flipped = np.random.random() < 0.5

    if flipped:
        real = torch.flip(real, [3])



fake = G(real)
b_fake = fake[:init_batch_a.size(0)]
idt_b = fake[init_batch_a.size(0):]

D_Loss(init_batch_b, b_fake).backward()
G_Loss_improved(init_batch_a, init_batch_b, idt_b, b_fake).backward()
#print(len(F.mlps))
F_opt = optim.Adam(F.parameters(),lr=lr,betas=(beta1,0.999))

del init_batch_a, init_batch_b, fake, b_fake, idt_b
    




print("Current CUDA memory allocated:", torch.cuda.memory_allocated())
torch.autograd.set_detect_anomaly(True)





def training_step(engine, data):
    """
        a: synthetic pizza
        b: real pizza
    """

    # Get data from batch

    a_real = data[0][0].to(device)
    b_real = data[1][0].to(device)

    real = torch.cat((a_real, b_real), dim=0)

    if flip:
        flipped = np.random.random() < 0.5

        if flipped:
            real = torch.flip(real, [3])
    fake = G(real)
    b_fake = fake[:a_real.size(0)]
    idt_b = fake[a_real.size(0):]

    # D loss
    D.train()
    D_opt.zero_grad()

    d_loss = D_Loss(b_real, b_fake)
    d_loss.backward()
    D_opt.step()

    D.eval()

    G_opt.zero_grad()
    #F_opt.zero_grad()
    loss_g = G_Loss_improved(a_real, b_real, idt_b, b_fake)

    loss_g.backward()
    G_opt.step()
    #F.step()
    
    del a_real, b_real, idt_b, b_fake
    return {
        "G_Loss": loss_g,
        "D_Loss": d_loss
    }

train_obj = Engine(training_step)


if load_model:

    to_save = {"G": G, 'D': D, 'F': F, "G_opt": G_opt, "D_opt":D_opt, "train_obj": train_obj}
    checkpoint_fp = f"./cut_improved_b2a/checkpoint_6.pt"
    checkpoint = torch.load(checkpoint_fp)
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    #F_opt.load_state_dict(checkpoint['F_opt'])


def init_weights(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


#@train_obj.on(Events.STARTED)
def init_w():
    G.apply(init_weights)
    D.apply(init_weights)
    F.apply(init_weights)
    



@train_obj.on(Events.ITERATION_COMPLETED)
def append_loss(engine):
    
    o = engine.state.output
    d_losses.append(o["D_Loss"])
    g_losses.append(o['G_Loss'])

    #print(len(list(engine.state.dataloader)))

 
def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img.cpu())
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)
    
def evaluation_step(engine, batch):
    with torch.no_grad():
        a_real = batch[0][0].to(device)
        b_real = batch[1][0].to(device)
        G.eval()
        b_fake = G(a_real)
        fake = interpolate(b_fake)
        real = interpolate(b_real)
        return fake, real


eval_obj = Engine(evaluation_step)


FIDScores = []
ISScores = []

@train_obj.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):

    """
    evaluator.run(zip(syn_test_dl, real_test_dl),max_epochs=1)


    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    FIDScores.append(fid_score)
    ISScores.append(is_score)

    

    print(f"Epoch [{engine.state.epoch}] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")

    """


    """if engine.state.iteration % 200 == 0:
        test_a = next(iter(a_test_dl))
        test_b = next(iter(b_test_dl))
        b_fake = G_A(test_a[0].to(device))
        a_fake = G_B(test_b[0].to(device))

        if not(os.path.isdir(f'./test_images/test_images_a2b_cycle/epoch_{engine.state.epoch}/')):
            os.mkdir(f'./test_images/test_images_a2b_cycle/epoch_{engine.state.epoch}/')

        if not(os.path.isdir(f'./test_images/test_images_b2a_cycle/epoch_{engine.state.epoch}/')):
            os.mkdir(f'./test_images/test_images_b2a_cycle/epoch_{engine.state.epoch}/')

        torchvision.utils.save_image(a_fake,f'./test_images/test_images_b2a_cycle/epoch_{engine.state.epoch}/iter_{engine.state.iteration}.png')
        torchvision.utils.save_image(b_fake,f'./test_images/test_images_a2b_cycle/epoch_{engine.state.epoch}/iter_{engine.state.iteration}.png')
"""

    print(f'Epoch {engine.state.epoch}')
    engine.state.dataloader = zip(b_train_dl, a_train_dl)
    #print(f'Dataloader length: {len(engine.state.dataloader)}')

to_save = {"G": G, 'D': D, 'F': F, "G_opt": G_opt, "D_opt":D_opt, 'F_opt':F_opt,"train_obj": train_obj}

gst = lambda *_: train_obj.state.epoch
handler = Checkpoint(
    to_save, save_handler=DiskSaver('./cut_improved_b2a/', create_dir=True, require_empty=False), n_saved=50, global_step_transform=gst
)
train_obj.add_event_handler(Events.EPOCH_COMPLETED, handler)


RunningAverage(output_transform=lambda x: x["G_Loss"]).attach(train_obj, 'G_Loss')
RunningAverage(output_transform=lambda x: x["D_Loss"]).attach(train_obj, 'D_Loss')

ProgressBar().attach(train_obj, metric_names=['G_Loss','D_Loss'])
#ProgressBar().attach(train_obj)
ProgressBar().attach(eval_obj)

def training(*args):
    train_obj.run(zip(b_train_dl, a_train_dl), max_epochs=50)


training()
