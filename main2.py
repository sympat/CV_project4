import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from Datasets import MyDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


datadir = './data/trainset'
img_size = 128

#set hyperparameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 25
batch_size = 30

#prepare data
dataset = MyDataset(
        root=datadir,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
)

idx = list(range(len(dataset)))
np.random.seed(1009)
np.random.shuffle(idx)
train_idx = idx[ : int(0.8 * len(idx))]
valid_idx = idx[int(0.8 * len(idx)) : ]

train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_set)
valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_set)

print("\n\tLoad Complete !")

#load pretrained model and reset fully connected layer
model = resnet50(pretrained = True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.cuda()

# Observe that all parameters are being optimized
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#start training
summary = SummaryWriter()
training_accuracy = []
validation_accuracy = []
for epoch in range(num_epochs):

    # training set -- perform model training
    epoch_training_loss = 0.0
    num_batches = 0

    for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function
        # split training data into inputs and labels
        inputs, labels = training_batch                              # 'training_batch' is a list
        # wrap data in 'Variable'
        inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())
        # Make gradients zero for parameters 'W', 'b'
        optimizer.zero_grad()
        # forward, backward pass with parameter update
        forward_output = model(inputs)
        loss = criterion(forward_output, labels)
        loss.backward()
        optimizer.step()
        # calculating loss
        epoch_training_loss += loss.item()
        num_batches += 1

    print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)

    #if epoch % 10 == 0:
    summary.add_scalar(tag='training loss', scalar_value=epoch_training_loss / num_batches, global_step=epoch)

    # calculate training set accuracy
    accuracy = 0.0
    num_batches = 0
    for batch_num, training_batch in enumerate(train_loader):        # 'enumerate' is a super helpful function
        num_batches += 1
        inputs, actual_val = training_batch
        # perform classification
        predicted_val = model(torch.autograd.Variable(inputs.cuda()))
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function
        predicted_val = predicted_val.cpu().data.numpy()
        predicted_val = np.argmax(predicted_val, axis=1)
        accuracy += accuracy_score(actual_val, predicted_val)
    training_accuracy.append(accuracy/num_batches)

    # calculate validation set accuracy
    accuracy = 0.0
    num_batches = 0
    for batch_num, validation_batch in enumerate(valid_loader):        # 'enumerate' is a super helpful function
        num_batches += 1
        inputs, actual_val = validation_batch
        # perform classification
        predicted_val = model(torch.autograd.Variable(inputs.cuda()))
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function
        predicted_val = predicted_val.cpu().data.numpy()
        predicted_val = np.argmax(predicted_val, axis=1)
        accuracy += accuracy_score(actual_val, predicted_val)
    print('accuracy:' + str(accuracy/num_batches))
    validation_accuracy.append(accuracy/num_batches)

torch.save(model.state_dict(), './save/params.ckpt')

epochs = list(range(num_epochs))

# plotting training and validation accuracies
fig1 = pyplot.figure()
pyplot.plot(epochs, training_accuracy, 'r')
pyplot.plot(epochs, validation_accuracy, 'g')
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy")
pyplot.show(fig1)