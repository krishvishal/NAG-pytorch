import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
from model import *
import time
from torchvision import models
from torch import optim


#Some constants for training, change them accordingly
size=224
batch_size = 32
lr = 1e-3


def add_perturbation(inp, perturbation):
	inp = inp + perturbation
	return inp

def add_pertubation2(inp, perturbation):
	k = inp.size()[0]
	for i in range(k):
		j = torch.LongTensor(1).random_(0, batch_size)
		temp = inp[i] + perturbation[j]
		temp = temp.view(1,3,size,size)
		if i==0:
			out = temp
		else:
			out = torch.cat((out, temp), 0)
	return out

def log_loss(prob_vec, adv_prob_vec, top_prob):
	size = prob_vec.size()[0]
	for i in range(size):
		if i==0:
			loss = adv_prob_vec[i][top_prob[i][0]]
		else:
			loss = loss + adv_prob_vec[i][top_prob[i][0]]

	mean = (loss/size)
	gen_loss = - torch.log(1 - mean)
	return gen_loss, mean



def validation_results(prob_adv, prob_real):
'''
	Helper function to calculate mismatches in the top index vector
	for clean and adversarial batch
	Parameters:
	prob_adv : Index vector for adversarial batch
	prob_real : Index vector for clean batch
	Returns:
	Number of mismatch and its percentage
'''
	nfool=0
	size = prob_real.size()[0]
	for i in range(size):
		if prob_real[i] != prob_adv[i]:
			nfool = nfool+1
	return nfool, 100*float(nfool)/size



def choose_net(network):
	if network == 'vgg16':
		net = models.vgg16(pretrained=true)

	if network == 'vgg19':
		net = models.vgg19(pretrained=true)
	if network == 'resnet50':
		net = models.resnet50(pretrained=true)
		for params in net.parameters():
			params.requires_grad = False
	if network == 'resnet152':
		net = models.resnet152(pretrained=true)
	if network == 'inceptionv3':
		net = models.inceptionv3(pretrained=true)

	return net

def scores(inp_val):
	net = choose_net('resnet50')
	out1 = net(inp_val)
	out = F.softmax(out1)
	_, topk = torch.topk(out, 1)
	return out, topk


def make_z(shape, minval, maxval):

	z = minval + torch.rand(shape) * (maxval - 1)
	return z

def save_checkpoint(state, epoch):
	ckpt_dir = 'home/vkv/NAG/ckpt/'	
    print("[*] Saving model to {}".format(ckpt_dir))

    filename = 'NAG' + '_ckpt.pth.tar'
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

model = Generator().cuda()
net = choose_net('resnet50')
net = net.cuda()

optimizer = optim.Adam(model.parameters() ,lr = lr)

for epoch in range(epochs):
	for i, images in enumerate(train_loader):
		images = images.cuda()
		z = make_z((model.batch_size, model.z_dim ), minval=-1., maxval=1.)
		z_ref = make_z((model.batch_size, model.z_dim ), minval=-1., maxval=1.)

		model.train()
		model.zero_grad()
		perturbations = model(z_ref, z)
		random_adv_batch = add_perturbation(images, perturbations)
		random_adv_batch2 = add_pertubation2(images, perturbations)
		v, topk = scores(images)
		v_adv, topk_adv = scores(random_adv_batch)
		v_adv2, _ = scores(random_adv_batch2)

		outputs = []
		def hook(module, input, output):
			outputs.append(output)
		net.layer4[0].conv2.register_forward_hook(hook)
		f1 = net(random_adv_batch)
		f2 = net(random_adv_batch2)
		f1_res4f = outputs[0]
		f2_res4f = outputs[1]

		feature_loss = -10*torch.mean(f1_res4f*f1_res4f - f2_res4f*f2_res4f)
		q1_loss, meanq1 = log_loss(v, v_adv, topk)
		q1_loss = q1_loss + feature_loss
		q1_loss.backward()
		optimizer.step()
		if i%30==0:
			np.save('running_perturbation.npy', perturbations)
		print ("{} {} {} {} {} {} {} {} {} {}".format("Epoch",epoch,"Iteration",i,"Log loss",q1_loss,"Mean",meanq1,"Feature loss",feature_loss))
		f = open('log_loss_imagenet.txt','a')
		f.write(str(q1_loss)+'\n')
		f.close()

		if i!=0 and i%100==0:
			total_fool = 0
			print ("{}".format("############### VALIDATION PHASE STARTED ################"))
			for j, images_val in enumerate(val_loader):
				images_val = images_val.cuda()
				z_val = make_z((model.batch_size, model.z_dim ), minval=-1., maxval=1.)
				z_ref_val = make_z((model.batch_size, model.z_dim ), minval=-1., maxval=1.)
				
				model.eval()
				perturbations_val = model(z_ref_val, z_val)
				random_adv_batch_val = add_pertubation(images_val, perturbations_val)
				random_adv_batch2_val = add_pertubation2(images_val, perturbations_val)
				v_val, topk_val = scores(images_val)
				v_adv_val, topk_adv_val = scores(random_adv_batch_val)
				nfool,foolr = validation_results(topk_val,topk_adv_val)
				total_fool = total_fool + nfool
			foolr = 100*float(total_fool)/(1000)
			print("{} {} {}".format("Fooling rate",foolr,total_fool))
			f = open('log_fool_rate_imagenet.txt', 'a')
			f.write(str(foolr)+'\n')
			f.close()
			print ("{}".format("############### VALIDATION PHASE ENDED ################"))


	save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()})
		


