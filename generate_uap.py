import scipy
from model import *
from trainer import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import scipy


def load_checkpoint():

	ckpt_dir = 'home/vkv/NAG/ckpt/'
	print("[*] Loading model from {}".format(ckpt_dir))

	filename = 'NAG' + '_ckpt.pth.tar'
	ckpt_path = os.path.join(ckpt_dir, filename)
	ckpt = torch.load(ckpt_path)

	# load variables from checkpoint
	model.load_state_dict(ckpt['state_dict'])
	
	print("[*] Loaded {} checkpoint @ epoch {} with best valid acc of {:.3f}".format(
				filename, ckpt['epoch'], ckpt['best_valid_acc']))

model = Generator().cuda()
net = choose_net('resnet50')
net = net.cuda()

load_checkpoint()
n=20

for i in range(n):
	z = make_z((model.batch_size, model.z_dim ), minval=-1., maxval=1.)
	z_ref = make_z((model.batch_size, model.z_dim ), minval=-1., maxval=1.)
	pert = model(z_ref, z)
	pert = pert.cpu().numpy()
	pert = np.transpose(pert, (0,2,3,1))
	np.save('perturbation' + str(i) + '.npy', pert[0])
	scipy.misc.imsave('perturbation' + str(i) + '.png', pert[0])
print("{} {}".format(n, "perturbations saved"))