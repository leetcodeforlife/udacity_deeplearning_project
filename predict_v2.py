#step 1 -- import packages 
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import time
import copy
import argparse
from PIL import Image
import json
import matplotlib.pyplot as plt


#step 2 -- define argparse variables
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, help='image path')
parser.add_argument('--checkpoint', type=str, help='model checkpoint file location')
parser.add_argument('--topk', type=int, help='top K classes')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--jason_dir', type=str, help='json file location')
args, _ = parser.parse_known_args()


#step 3 -- load model from checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch=checkpoint['arch']
    if arch=='vgg19':
        model=models.vgg19(pretrained=True)
    elif arch=='vgg13':
        model=models.vgg13(pretrained=True)
    else:
        print("error, arch does not available in this version")
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    return model

model=load_checkpoint(args.checkpoint)

#step 4 -- load json file # 
json_file='{}/cat_to_name.json'.format(args.jason_dir)

with open(json_file,'r') as f:
    cat_to_name = json.load(f)
    
#step 5 -- input processing#
def process_image(image):
    im=Image.open(image)
    im_transform=transforms.Compose([transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])
    std_image=np.array(im_transform(im).float())
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    std_image = (np.transpose(std_image, (1, 2, 0)) - mean)/std    
    std_image = np.transpose(std_image, (2, 0, 1))
    return std_image


class_to_idx=model.class_to_idx

#step 6 -- make prediction 
def predict(image_path, model, topk=5,gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu ==True:
        im=model.cuda()
    else:
        im=model
    im=process_image(image_path)
    im = Variable(torch.FloatTensor(im))
    im = im.unsqueeze(0) 
    if gpu ==True:
        im=im.cuda()
    outputs = model(im)
    #from predicted index to classes 
    prediction_index=torch.topk(outputs, topk)
    prediction_classes=[]
    for i in Variable(prediction_index[1], requires_grad=False).cpu().numpy()[0]:
        classname=list(class_to_idx.keys())[list(class_to_idx.values()).index(i)]
        prediction_classes.append(classname)
    final_topk_prediction=[np.exp(Variable(prediction_index[0], requires_grad=False).cpu().numpy()[0]).tolist() #prob prediction
                          ,prediction_classes] #class prediction
    return final_topk_prediction

def sanity_check(image,model):
    probs, classes=predict(image,model,args.topk,args.gpu)
    classes_name=[]
    print('Below are top {} possible flowers'.format(args.topk))
    for i in classes:
        print(cat_to_name.get(i), probs[classes.index(i)])

sanity_check(args.image_dir,model)

#to run 
#python predict.py --image_dir flowers/train/10/image_07093.jpg --checkpoint checkpoint_model_trainpy.pth --topk 10 --gpu True --jason_dir /home/workspace/aipnd-project