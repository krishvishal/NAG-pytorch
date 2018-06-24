import json
import urllib
from tqdm import *
import argparse
import os, sys
from random import shuffle
import urllib.request
import cv2
import numpy as np


parser = argparse.ArgumentParser(description="Download images to given folder")
parser.add_argument("-c", type = str, default = 'car_wheel', help = "The class whose images will be downloaded")
parser.add_argument("-n", type = str, default = "10", help = "The number of images to be downloaded (with auto upper limit)")
parser.add_argument("-o", type = str, help = "output folder for downloaded images")
args = parser.parse_args()

base_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid='

with open('imagenet_class_index.json', 'r') as f:
	class_index = json.loads(f.read())
class_to_wnid = {}
for i in range(1000):
	wnid, class_name = class_index[str(i)]
	class_to_wnid[class_name.lower()] = wnid
class_name = args.c
class_wnid = str(class_to_wnid[class_name])

class_wnid = list(class_to_wnid.values())

for j in range(len(class_wnid)): 
	link = base_link + class_wnid[j]
	link_image_urls = urllib.request.urlopen(link).read().decode()

	# if not os.path.exists('/home/vkv/imagenet'):
	# 	os.makedirs('home/vkv/imagenet')
	print('[LOG]: WordNet ID: {}'.format(class_wnid[j]))


	# The code doesn't consider the cases where the num_images > images in the dataset
	# Take Note

	num_images = int(args.n)
	pic_num = 1

	for i in link_image_urls.split('\n'):
		try:
			print(i)
			urllib.request.urlretrieve(i, '/home/vkv/imagenet/' + class_wnid[j] + str(pic_num) + '.jpg')
			im = cv2.imread('/home/vkv/imagenet/' + class_wnid[j] + str(pic_num) + '.jpg')
			pic_num += 1
			if np.any(im == None):
				pic_num -= 1
				os.remove('home/vkv/imagenet/' + class_wnid[j] + str(pic_num) + '.jpg')
			
			if (pic_num == num_images+1):
				break

		except Exception as e:
			print(str(e))
