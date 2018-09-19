#step 1 -- import packages 
import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import time
import copy
import argparse



#step 2 -- define argparse variables
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
args, _ = parser.parse_known_args()


#step 3 -- create dataloaders based on arg input

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],   #RGB mean used to normalize
                                                            std=[0.229, 0.224, 0.225])]) #RGB std used to normalize

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data=datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

class_to_idx=train_data.class_to_idx


#step 4 -- load pre-trained model
def load_model(arch='vgg13'):
    if arch=='vgg13':
        model = models.vgg13(pretrained=True)
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError('Only vgg13 and vgg13 are available as network architecture')
    return model

model=load_model(args.arch)

#step 5 -- train classifier on the pre-trained model
def train_model(epochs=3,learing_rate=0.001, hidden_units=4096,gpu=True):
    for param in model.parameters():
        param.requires_grad = False
    if args.epochs:
        epochs=args.epochs
    if args.learning_rate:
        lr=args.learning_rate
    if args.hidden_units:
        hidden_units=args.hidden_units
    if args.gpu:
        gpu=args.gpu 
    if args.arch=='vgg13' or args.arch=='vgg19':
        arch=args.arch
    else:
        arch='vgg13'
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    print_every = 40
    steps = 0
    

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu ==True:
                model.to('cuda')
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0
                
    checkpoint_file=args.checkpoint
    checkpoint = {'epochs': epochs + 1,
                  'arch':arch,
                  'hidden_units':hidden_units,
                  'learning_rate':lr,
                  'class_to_idx':class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, checkpoint_file)
        
train_model()

#step 6 -- validation 
# Validation on the validation  dataset
def validation_model(gpu=True):
  if args.gpu:
    gpu=args.gpu 
  correct = 0
  total = 0
  with torch.no_grad():
      for data in validloader:
          images, labels = data
          if gpu ==True:
              model.to('cuda')
              images, labels = images.to('cuda'), labels.to('cuda')
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 valid images: %d %%' % (100 * correct / total))

validation_model()

#to run 
#python train.py --data_dir flowers --gpu True --epochs 4 --arch vgg13 --learning_rate 0.001 --hidden_units 4096 --checkpoint checkpoint_model_trainpy_2.pth



