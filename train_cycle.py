
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
from models.generator import Generator
from models.discriminator import Discriminator
from models.img_pool import ImgPool
import ignite.distributed as idist
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import FID, InceptionScore, RunningAverage
import PIL.Image as Image
from ignite.contrib.handlers import ProgressBar
import logging
from ignite.engine import Engine, Events
import ignite
from pizzamaker.syntheticPizza import SyntheticPizzaDataset
import torch.nn.init as init


ignite.utils.manual_seed(999)
ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)

torch.cuda.empty_cache()
#device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
#print(device)
data_root = './data/pizza/pizza/'
bs = 5
workers = 2
image_size = (256,256)
device = idist.device()

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

    syn_live_train_dl = idist.auto_dataloader(syn_live_train_ds, batch_size=bs, drop_last=True, shuffle=True)


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

    synprerec_train_dl = idist.auto_dataloader(synprerec_train_ds, batch_size=bs, drop_last=True, shuffle=True)

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


    syn_test_dl = idist.auto_dataloader(syn_eval_ds, batch_size=bs,drop_last=True, shuffle=True)


    return synprerec_train_dl, syn_test_dl



def get_real_data():
    real_train_ds = dset.ImageFolder(root=data_root+'real_train/',
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    real_train_dl = idist.auto_dataloader(real_train_ds, batch_size=bs,drop_last=True, shuffle=True)

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

    return real_train_dl, real_test_dl



a_train_dl= get_syn_live_data()
b_train_dl, b_test_dl = get_real_data()


G_A = idist.auto_model(Generator())# translates images a to b
G_B = idist.auto_model(Generator())# translates images b to a
D_A = idist.auto_model(Discriminator()) # determines whether an image is in a or not
D_B = idist.auto_model(Discriminator()) # determines whether an image is in b or not

#print(summary(G_A, input_size=(bs, 3, 256, 256)))
bce_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

"""
    calc D(A)

"""

def discriminator_loss(real, fake): 
    
    l_real = bce_loss(real,torch.ones_like(real).float())
    l_gen = bce_loss(fake, torch.zeros_like(fake).float())

    return 0.5*(l_real + l_gen)



def generator_loss(pred, is_real):
    
    """
        Generator Loss:
            - BCEWithLogitsLoss
    """
    target = torch.tensor(int(is_real), device=device).expand_as(pred).float()
    return bce_loss(pred, target)

def cycle_consistency_loss(real, cycled):
    
    return l1_loss(cycled, real)

def identity_loss(real, copy):
    return l1_loss(copy, real)

lr = 2e-4
beta1 = 0.5
G_A_opt = idist.auto_optim(optim.Adam(G_A.parameters(),lr=lr,betas=(beta1,0.999)))
G_B_opt = idist.auto_optim(optim.Adam(G_B.parameters(),lr=lr,betas=(beta1,0.999)))
D_A_opt = idist.auto_optim(optim.Adam(D_A.parameters(),lr=lr,betas=(beta1,0.999)))
D_B_opt = idist.auto_optim(optim.Adam(D_B.parameters(),lr=lr,betas=(beta1,0.999)))

img_pool_a = ImgPool(size=50)
img_pool_b = ImgPool(size=50)

g_losses = []
d_a_losses = []
d_b_losses = []


print("Current CUDA memory allocated:", torch.cuda.memory_allocated())
#torch.autograd.set_detect_anomaly(True)


def training_step(engine, data):
    """
        a: synthetic pizza
        b: real pizza
    """

    # Set models to train mode
    
    """G_A.eval()
    G_B.eval()
    D_A.train()
    D_B.train()"""

    
    # Get data from batch

    a_real = data[0][0].to(device)
    b_real = data[1][0].to(device)

   

    # Generate images
    a_fake = G_B(b_real)
    b_cycled = G_A(a_fake)
    b_fake = G_A(a_real)
    a_cycled = G_B(b_fake)
    



    D_A.train()
    D_B.train()
    # Update D_A weights
    D_A_opt.zero_grad()

    disc_a_real = D_A(a_real) # 
    a_temp_fake = img_pool_a.query(a_fake)
    disc_a_fake = D_A(a_temp_fake.detach())

    disc_a_real_loss = generator_loss(disc_a_real, True)
    #print('calculating disc a fake loss')
    disc_a_fake_loss = generator_loss(disc_a_fake, False)
    #print('summing')
    disc_a_loss = 0.5*(disc_a_real_loss + disc_a_fake_loss)
    #print(disc_a_loss)
    disc_a_loss.backward()
    D_A_opt.step()

    # Updating discriminator for images from B
    D_B_opt.zero_grad()
    disc_b_real = D_B(b_real)

    b_temp_fake = img_pool_b.query(b_fake)

    disc_b_fake = D_B(b_temp_fake.detach())
    
    disc_b_real_loss = generator_loss(disc_b_real, True)
    disc_b_fake_loss = generator_loss(disc_b_fake, False)
    disc_b_loss = 0.5*(disc_b_real_loss + disc_b_fake_loss)
    disc_b_loss.backward()
    D_B_opt.step()

    G_A_opt.zero_grad()
    G_B_opt.zero_grad()
    
    """G_A.train()
    G_B.train()
    D_B.eval()
    D_A.eval()"""

    D_A.eval()
    D_B.eval()

    gen_a_loss = generator_loss(D_B(b_fake), True)
    gen_b_loss = generator_loss(D_A(a_fake), True)

    cyc_a_loss = cycle_consistency_loss(a_real, a_cycled) * 5
    cyc_b_loss =  cycle_consistency_loss(b_real, b_cycled) * 5
    
    a_idt = G_A(b_real)
    b_idt = G_B(a_real)

    idt_a_loss = identity_loss(a_idt, b_real)*10 # 0.5 default value
    idt_b_loss = identity_loss(b_idt, a_real)*10


    total_gen_loss = gen_a_loss + gen_b_loss + cyc_a_loss + cyc_b_loss + idt_a_loss + idt_b_loss

    total_gen_loss.backward()
    G_A_opt.step()
    G_B_opt.step()

    
    del a_real, b_real, a_fake, b_fake
    return {
        "G_Loss": total_gen_loss,
        "G_A_Loss": gen_a_loss,
        "G_B_Loss": gen_b_loss,
        "D_A_Loss": disc_a_loss,
        "D_B_Loss": disc_b_loss
    }

train_obj = Engine(training_step)

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
    G_A.apply(init_weights)
    G_B.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)



@train_obj.on(Events.ITERATION_COMPLETED)
def append_loss(engine):
    
    o = engine.state.output
    d_a_losses.append(o["D_A_Loss"])
    d_b_losses.append(o["D_B_Loss"])
    g_losses.append(o['G_Loss'])

    #print(len(list(engine.state.dataloader)))


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
    engine.state.dataloader = zip(a_train_dl, b_train_dl)
    #print(f'Dataloader length: {len(engine.state.dataloader)}')

to_save = {"train_obj":train_obj, "G_A": G_A, 'G_B': G_B, 'D_A': D_A, 'D_B': D_B, "G_A_opt":G_A_opt, "G_B_opt":G_B_opt, "D_A_opt":D_A_opt, "D_B_opt":D_B_opt}


to_load = to_save
checkpoint_fp = "./cycle_models_live2/checkpoint_18.pt"
checkpoint = torch.load(checkpoint_fp)
Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

gst = lambda *_: train_obj.state.epoch
handler = Checkpoint(
    to_save, save_handler=DiskSaver('./cycle_models_live2', create_dir=True, require_empty=False), n_saved=50, global_step_transform=gst
)
train_obj.add_event_handler(Events.EPOCH_COMPLETED, handler)


RunningAverage(output_transform=lambda x: x["G_A_Loss"]).attach(train_obj, 'G_A_Loss')
RunningAverage(output_transform=lambda x: x["D_A_Loss"]).attach(train_obj, 'D_A_Loss')
RunningAverage(output_transform=lambda x: x["G_B_Loss"]).attach(train_obj, 'G_B_Loss')
RunningAverage(output_transform=lambda x: x["D_B_Loss"]).attach(train_obj, 'D_B_Loss')
ProgressBar().attach(train_obj, metric_names=['G_A_Loss','G_B_Loss','D_A_Loss', 'D_B_Loss'])
#ProgressBar().attach(train_obj)


def training(*args):
    train_obj.run(zip(a_train_dl, b_train_dl), max_epochs=50)


with idist.Parallel(backend='nccl') as parallel:
    parallel.run(training)

#train_obj.run(zip(a_train_dl, b_train_dl), max_epochs=50)