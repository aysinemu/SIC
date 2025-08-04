import argparse
import os
import numpy as np
import math
import json
import itertools
import shutil
import lpips
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from utils.horse2zebra import HorseZebraDatasetFromCSV
from utils.mnistm import MNISTM
from utils.Classifier import Classifier
from utils.Generator import Generator
from utils.Discriminator import Discriminator
from utils.SelfAttention import SelfAttention
from utils.ADAugment import ADAugment
from utils.ConditionalDiffusionModel import ConditionalDiffusionModel, get_beta_schedule, forward_diffusion, sample_conditional_diffusion
from config import opt

if os.path.exists("imagess.json"):
    os.remove("imagess.json")
if os.path.exists("imagess"):
    shutil.rmtree("imagess")
os.makedirs("imagess", exist_ok=True)
# if os.path.exists("diffusion_samples"):
#     shutil.rmtree("diffusion_samples")
# os.makedirs("diffusion_samples", exist_ok=True)


# Calculate output of image discriminator (PatchGAN)
patch = int(opt.img_size / 2 ** 4)
patch = (1, patch, patch)

cuda = True if torch.cuda.is_available() else False
device = torch.device(f"cuda:{opt.n_gpu}" if torch.cuda.is_available() else "cpu")

if cuda:
    torch.cuda.set_device(opt.n_gpu) 

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def entropy_minimization_loss(logits):
    """
    Tính entropy loss cho đầu ra của mô hình (logits).
    logits: Tensor [B, C, H, W] hoặc [B, C]
    """
    p = torch.softmax(logits, dim=1)
    log_p = torch.log_softmax(logits, dim=1)
    entropy = -torch.sum(p * log_p, dim=1)  # [B, H, W] hoặc [B]
    return entropy.mean()

def semantic_consistency_loss(logits_src, logits_tgt):
    """
    So sánh logits của ảnh source và ảnh chuyển domain bằng KL Divergence.
    logits_src, logits_tgt: Tensor [B, C, H, W] hoặc [B, C]
    """
    p_src = torch.softmax(logits_src, dim=1)
    log_p_tgt = torch.log_softmax(logits_tgt, dim=1)
    return F.kl_div(log_p_tgt, p_src, reduction='batchmean')

def preprocess(img):
    return img * 2 - 1

def colorize_mnist(img):
    img = transforms.Resize(opt.img_size)(img)
    img = transforms.ToTensor()(img)
    r, g, b = torch.rand(3)
    img_rgb = torch.cat([
        img * r,
        img * g,
        img * b
    ], dim=0)
    return transforms.Normalize([0.5]*3, [0.5]*3)(img_rgb)

# Loss function
adversarial_loss = torch.nn.MSELoss()
task_loss = torch.nn.CrossEntropyLoss()
loss_fn = lpips.LPIPS(net='vgg').to(device)

# Loss weights
lambda_adv = 1
lambda_style = 1
lambda_task = 1
lambda_semantic = 1
lambda_cycle = 1
lambda_idt = 1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()
betas = get_beta_schedule().cuda()
diffusion_model = ConditionalDiffusionModel(channels=opt.channels, img_size=opt.img_size).cuda()
optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=opt.lr)

if cuda:
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    adversarial_loss.cuda()
    task_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
classifier.apply(weights_init_normal)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
dataloader_A = torch.utils.data.DataLoader(
    datasets.MNIST(
        "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/Dat/Teo/SIC/PyTorch-GAN/data/mnist",
        train=True,
        download=False,
        transform=colorize_mnist,
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

dataloader_C = torch.utils.data.DataLoader(
    datasets.MNIST(
        "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/Dat/Teo/SIC/PyTorch-GAN/data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# os.makedirs("../../data/mnistm", exist_ok=True)
dataloader_B = torch.utils.data.DataLoader(
    MNISTM(
        "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/Dat/Teo/SIC/PyTorch-GAN/data/mnistm",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.AdamW(
    itertools.chain(generator.parameters(), classifier.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-5
)
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# Keeps 100 accuracy measurements
task_performance = []
target_performance = []
log_data = []
best = float('inf')
step = 0 
ada = ADAugment(prob=0.0)

# for epoch in tqdm(range(500)):  # Pretrain 50 epochs
#     for imgs_A, _ in dataloader_C:
#         imgs_A = imgs_A.cuda()
#         t = torch.randint(0, 1000, (imgs_A.size(0),), device=device)
        
#         x_t, noise = forward_diffusion(imgs_A, t, betas, device)
#         pred_noise = diffusion_model(x_t, t, imgs_A)
        
#         loss = nn.MSELoss()(pred_noise, noise)
#         optimizer.zero_grad()
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
#         optimizer.step()
#     print(f"Epoch {epoch} done. Loss: {loss.item():.4f}")

#     with torch.no_grad():
#         sample_imgs = sample_conditional_diffusion(
#             diffusion_model,
#             imgs_A[:5],  
#             opt.channels,
#             opt.img_size,
#             betas,
#             device
#         )
#         save_image(sample_imgs, f"diffusion_samples/epoch_{epoch}.png", normalize=True)
# torch.save(diffusion_model.state_dict(), 'diffusion_model.pth')

diffusion_model.load_state_dict(torch.load('diffusion.pth'))
diffusion_model.eval()

for epoch in tqdm(range(opt.n_epochs)):
    for i, ((imgs_A, labels_A), (imgs_B, labels_B), (imgs_C, labels_C)) in tqdm(enumerate(zip(dataloader_A, dataloader_B, dataloader_C))):
        
        imgs_A = imgs_A.cuda()
        batch_size_A = imgs_A.size(0)
        imgs_C = imgs_C.cuda()
        batch_size_C = imgs_C.size(0)

        # Adversarial ground truths
        valid_A = Variable(torch.tensor(1.0, dtype=torch.float, device=device).expand(batch_size_A, *patch), requires_grad=False)
        fake_A = torch.tensor(0.0, dtype=torch.float, device=device).expand(batch_size_A, *patch)
        valid_C = Variable(torch.tensor(1.0, dtype=torch.float, device=device).expand(batch_size_C, *patch), requires_grad=False)
        fake_C = torch.tensor(0.0, dtype=torch.float, device=device).expand(batch_size_C, *patch)

        # Configure input
        imgs_A = Variable(imgs_A.type(FloatTensor).expand(batch_size_A, opt.channels, opt.img_size, opt.img_size))
        labels_A = Variable(labels_A.type(LongTensor))
        imgs_C = Variable(imgs_C.type(FloatTensor).expand(batch_size_C, opt.channels, opt.img_size, opt.img_size))
        labels_C = Variable(labels_A.type(LongTensor))
        imgs_B = Variable(imgs_B.type(FloatTensor))

        # diffusion_model.load_state_dict(torch.load('best_diff.pth'))
        # diffusion_model.eval()

        # with torch.no_grad():
        #     diffused_imgs = sample_conditional_diffusion(diffusion_model, imgs_C, opt.channels, opt.img_size, betas, device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise
        z_A = torch.rand(batch_size_A, opt.latent_dim, dtype=torch.float, device=device) * 2 - 1
        # z_A = torch.randn(batch_size_A, opt.latent_dim).cuda()

        # Generate a batch of images
        if epoch < 400:
            fake_B = generator(imgs_A, z_A)
        elif epoch >= 400 and epoch < 800:
            fake_B = generator(imgs_C, z_A)
        else:
            with torch.no_grad():
                diffused_imgs = sample_conditional_diffusion(diffusion_model, imgs_C, opt.channels, opt.img_size, betas, device)
            fake_B = generator(diffused_imgs, z_A)

        # Perform task on translated source image
        label_pred_B = classifier(fake_B)

        # Calculate the task loss
        # CrossEntropyLoss = (task_loss(label_pred_B, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2

        if epoch < 400:
            CrossEntropyLoss = (task_loss(label_pred_B, labels_A) + task_loss(classifier(imgs_A), labels_A)) / 2
        elif epoch >= 400 and epoch < 800:
            CrossEntropyLoss = (task_loss(label_pred_B, labels_C) + task_loss(classifier(imgs_C), labels_C)) / 2
        else:
            CrossEntropyLoss = (task_loss(label_pred_B, labels_C) + task_loss(classifier(diffused_imgs), labels_C)) / 2
            
        semantic_loss = semantic_consistency_loss(
            classifier(imgs_B).detach(),
            classifier(fake_B)
        )
        
        lpips_loss = loss_fn(preprocess(fake_B), preprocess(imgs_B)).mean()
        
        # if epoch < 400:
        #     cycle_loss = F.l1_loss(generator(fake_B, z_A), imgs_A)
        # elif epoch >= 400 and epoch < 800:
        #     cycle_loss = F.l1_loss(generator(fake_B, z_A), diffused_imgs)
        # else:
        #     cycle_loss = F.l1_loss(generator(fake_B, z_A), imgs_C)
        
        # cycle_loss = F.l1_loss(generator(fake_B, z_A), imgs_A)  # Zebra -> Horse -> Zebra

        # idt_loss = F.l1_loss(generator(imgs_B, z_A), imgs_B)  # Horse -> Horse should be unchanged
        
        # Loss measures generator's ability to fool the discriminator
        g_loss = (lambda_adv * adversarial_loss(discriminator(fake_B), valid_A) 
                + lambda_task * CrossEntropyLoss 
                + semantic_loss * lambda_semantic 
                + lpips_loss * lambda_style)
        
        g_loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(
        #     itertools.chain(generator.parameters(), classifier.parameters()),
        #     max_norm=1.0
        # )
        
        optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        if step % opt.n_critic == 0:
            
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss_A = adversarial_loss(discriminator(imgs_B), valid_A)
            # real_loss_AA = adversarial_loss(discriminator(imgs_A), fake_A)
            fake_loss_B = adversarial_loss(discriminator(fake_B.detach()), fake_A)
            
            d_loss = (real_loss_A + fake_loss_B) / 2

            d_loss.backward()
            optimizer_D.step()

            # # Cho Discriminator 1
            # real_accuracy = (real_loss_A > 0.7).float().mean().item()
            # if real_accuracy > 0.6:
            #     ada.prob = min(ada.prob + 0.01, 1.0)
            # else:
            #     ada.prob = max(ada.prob - 0.01, 0.0)
        
        step += 1
        # ---------------------------------------
        #  Evaluate Performance on target domain
        # ---------------------------------------

        # Evaluate performance on translated Domain A
        if epoch < 400:
            acc = np.mean(np.argmax(label_pred_B.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        else:
            acc = np.mean(np.argmax(label_pred_B.data.cpu().numpy(), axis=1) == labels_C.data.cpu().numpy())
        # acc = np.mean(np.argmax(label_pred_B.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
        task_performance.append(acc)
        if len(task_performance) > 100:
            task_performance.pop(0)

        # Evaluate performance on Domain B
        pred_B = classifier(imgs_B)
        target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.numpy())
        target_performance.append(target_acc)
        if len(target_performance) > 100:
            target_performance.pop(0)

        if target_acc >= best:
            best = target_acc
            
            torch.save(generator.state_dict(), 'acc.pth')            
            print(f"Saved model at epoch {epoch} with best g_loss = {best:.4f}")
        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)] [semantic_loss: %f] [lpips_loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader_A),
                d_loss.item(),
                g_loss.item(),
                100 * acc,
                100 * np.mean(task_performance),
                100 * target_acc,
                100 * np.mean(target_performance),
                semantic_loss.item(),
                lpips_loss.item(),
            )
        )

        log_data.append({
            "epoch": epoch,
            "batch": i,
            "CrossEntropyLoss": CrossEntropyLoss.item(),
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "real_loss_A": real_loss_A.item(),
            "fake_loss_B": fake_loss_B.item(),
            "acc": float(acc),
            "target_acc": float(target_acc),
            "task_performance": float(np.mean(task_performance)),
            "target_acc": float((target_acc)),
            "target_performance": float((np.mean(target_performance))),
            "semantic_loss": semantic_loss.item(),
            "lpips_loss": lpips_loss.item(),
        })
        
        batches_done = len(dataloader_A) * epoch + i
        if batches_done % opt.sample_interval == 0:
            if epoch < 400:
                sample_A = torch.cat((imgs_A.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            elif epoch >= 400 and epoch < 800:
                sample_A = torch.cat((imgs_C.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            else:
                sample_A = torch.cat((diffused_imgs.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            # sample_A = torch.cat((imgs_A.data[:5], diffused_imgs.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
            save_image(sample_A, "imagess/ST%d.png" % batches_done, nrow=int(math.sqrt(batch_size_A)), normalize=True)
    torch.save(generator.state_dict(), 'last.pth')   
    with open("imagess.json", "a") as f:
        f.write(json.dumps(log_data[-1], indent=2) + ",\n")
torch.save(generator.state_dict(), 'best.pth')  