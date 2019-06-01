import argparse, pprint, os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import alexnet, resnet18, resnet50
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from Datasets import MyDataset

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--is_train', type=str2bool, default=True, help='whether or not is train mode')
parser.add_argument('--ckptdir', type=str, default='./save', help='dir of saving checkpoints')
parser.add_argument('--summarydir', type=str, default='./summary', help='dir of saving summary')
parser.add_argument('--traindir', type=str, default='./data/trainset', help='dir of train images')
parser.add_argument('--testdir', type=str, default='./data/sample', help='dir of test images')
parser.add_argument('--epoch', type=int, default=10, help='size of epochs')
parser.add_argument('--batch', type=int, default=100, help='size of batches')
parser.add_argument('--lr', type=float, default=2e-4, help='size of learning rate')
parser.add_argument('--img_size', type=int, default=128, help='size of image')
FLAGS = parser.parse_args()


def main():
    # Print settings
    print('\n\t#### Parameter setting ####')
    print("\t%s: %s" % ('device', device))
    for FLAG, value in vars(FLAGS).items():
        print("\t%s: %s" % (FLAG, str(value)))

    
    if FLAGS.is_train:
        # Make directory if summarydir is not exist
        if not os.path.exists(FLAGS.summarydir):
            os.makedirs(FLAGS.summarydir)
        # Do train
        train(
            ckptdir=FLAGS.ckptdir, 
            traindir=FLAGS.traindir, 
            epochs=FLAGS.epoch, 
            batches=FLAGS.batch,
            lr=FLAGS.lr,
            img_size=FLAGS.img_size
        )
    else:
        # Do test
        test(
            ckptdir=FLAGS.ckptdir,
            testdir=FLAGS.testdir,
            epochs=FLAGS.epoch, 
            batches=FLAGS.batch,
            lr=FLAGS.lr
        )


def train(ckptdir, traindir, epochs, batches, lr, img_size):
    print("\n\tStart train ...")

    # Load data
    print("\n\tLoad Dataset ...")
    dataset = MyDataset(
        root=traindir,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    )
    # train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True)
    # total_step = len(train_loader)

    idx = list(range(len(dataset)))
    np.random.seed(1009)
    np.random.shuffle(idx)
    train_idx = idx[ : int(0.8 * len(idx))]
    valid_idx = idx[int(0.8 * len(idx)) : ]

    train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batches, sampler=train_set)
    valid_loader = DataLoader(dataset, batch_size=batches, sampler=valid_set)
    total_step = len(train_loader)

    print("\n\tLoad Complete !")

    # Define model
    model_name = 'resnet50'
    model = resnet50(pretrained = True)
    num_features = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(num_features, 3)
    model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # Define some variables for summary 
    loss_list = []
    epoch_list = []
    training_accuracy = []
    validation_accuracy = []

    for epoch in range(epochs):

        epoch_loss = 0.0

        # Train 
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # print the loss per epoch
        print ('\tEpoch [{}/{}], Loss: {:.4f}' 
            .format(epoch+1, epochs, epoch_loss / total_step))

        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader): 
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # print train accuracy per epoch
        print ('\tEpoch [{}/{}], Train accuracy: {:.4f}' 
            .format(epoch+1, epochs, correct / total))
        training_accuracy.append(correct / total)

    # Summarize train result 
    plt.plot(epoch_list, loss_list, label="%s_e%d_b%d_lr%f" % (model_name, epochs, batches, lr))
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.show()

    # Save trained model
    name = "train_%s_e%d_b%d_lr%f" % (model_name, epochs, batches, lr) 
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    torch.save(model.state_dict(), os.path.join(ckptdir, name+'.ckpt'))

    print("\tEnd train ...")


def test(ckptdir, testdir, epochs, batches, lr):
    print("\n\tStart test ...")

    # Load data
    print("\n\tLoad Dataset ...")
    test_dataset = MyDataset(
        root=testdir,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    )
    test_loader = DataLoader(test_dataset, batch_size=batches)
    total_step = len(test_loader)
    print(total_step)
    print("\n\tLoad Complete !")

    # Define model
    model_name = 'resnet50'
    model = resnet50(pretrained = True)
    num_features = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(num_features, 3)
    model.cuda()

    name = "train_%s_e%d_b%d_lr%f" % (model_name, epochs, batches, lr) 
    ckpt_path = os.path.join(ckptdir, name+'.ckpt')
    if os.path.isfile(ckpt_path):
        print("\tLoad model SUCCESS ^_^")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("\tLoad model FAILED >_<! trained model must be loaded for testing")
        return
    
    output_file = open('output.txt', 'w', encoding='utf-8')

    # Test
    for images, names in test_loader:
        images = images.to(device)
        names = names.to(device)
    
        # Forward pass
        outputs = model(images)
        print(outputs.size())
        return
        predicted_value, _ = torch.max(outputs.data, 1)



    print("\tEnd test ...")


if __name__ == '__main__':
    main()