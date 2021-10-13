import torch
from torch import optim
from torch.utils import data
from torch import nn

import numpy as np
from train_helpers import normalize, get_loss_weights, load_losses, save_losses
from datasets.normal_edf import Normal_Dataset

from models import Augmenting_Model
import os.path as op
import os

from tqdm import tqdm

import argparse

from torch.utils.tensorboard import SummaryWriter

root = op.dirname(__file__)
saved_models_dir = op.join(root, 'saved_models')
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

    
#Tensorboard    
writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Self Supervised EEG model training.')
parser.add_argument('--batch-size', '-b', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--number-workers', '-n', type=int, default=16, metavar='N', help='number of works to load dataset (default: 16)')
parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='L', help='initial learning rate (default: 1e-3)')
parser.add_argument('--weight-decay', type=float, default=0., metavar='W', help='weight decay (default: 0)')

parser.add_argument('--train-data', type=str, default='./datasets/demo_train.txt', help='dataset path for training (default: ./datasets/demo_train.txt)')
parser.add_argument('--test-data', type=str, default='./datasets/demo_test.txt', help='dataset path for testing (default: ./datasets/demo_test.txt)')
parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: None)')
parser.add_argument('--load-losses', type=str, default='', help='path to load latest checkpoint losses (default: None)')
parser.add_argument('--save-losses', type=str, default='', help='path to save new checkpoint losses (default: None)')
parser.add_argument('--save-name', type=str, default='checkpoint', help='path to save the final model')


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set the random seed manually for reproducibility
torch.manual_seed(args.seed)


# TO DO: read args to set parameters
def train_aug_ssl(train_loader, test_loader, model, n_epochs, lr, resume=False):

    new_train_losses, new_test_losses, train_accs, test_accs = _train_test_loop(model, train_loader, test_loader, epochs=n_epochs, lr=lr)
    
    if resume:
        train_losses, test_losses = load_losses(saved_models_dir, args.load_losses)
    else:
        train_losses = []
        test_losses = []
        
    train_losses.extend(new_train_losses)
    test_losses.extend(new_test_losses)

    save_losses(train_losses, test_losses, saved_models_dir, args.save_losses)

    return train_losses, test_losses, train_accs, test_accs, model


def _train_test_loop(model, train_loader, test_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, epochs+1):
        train_loss, train_acc = _train_loss(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_loss, test_acc = _eval_loss(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch}, Train loss {train_loss:.4f}, Train Accuracy {train_acc: .4f}, \
                Test loss {test_loss:.4f}, Test Accuracy {test_acc:.4f}')
        
        scheduler.step()
        
        # save model every 25 epochs
        if epoch % 25 == 0:
            torch.save(model.state_dict(), op.join(root, 'saved_models', args.save_name + '_epoch{}.pt'.format(epoch)))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/test', test_acc, epoch)
        
        writer.flush()
            
    #Save trained model
    torch.save(model.state_dict(), op.join(root, 'saved_models', args.save_name + '.pt'))
    return train_losses, test_losses, train_accs, test_accs


def _train_loss(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    epoch_train_acc = 0.0
    
    for i, data in enumerate(tqdm(train_loader)):
        x1, x2, y = data
        x1, x2, y = x1.to(device, dtype=torch.float), x2.to(device, dtype=torch.float), y.to(device)
        optimizer.zero_grad()
        out = model(x1, x2)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        y_pred = torch.round(torch.sigmoid(out))
        correct = (y_pred == y).sum().float()
        acc = correct/y.shape[0]
        acc = torch.round(acc * 100)
        epoch_train_acc += acc

    epoch_train_loss = total_loss / len(train_loader)
    epoch_train_acc = epoch_train_acc / len(train_loader)
    
    return epoch_train_loss, epoch_train_acc


'''
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc
'''

def _eval_loss(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    epoch_test_acc = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            x1, x2, y = data           
            x1, x2, y = x1.to(device, dtype=torch.float), x2.to(device, dtype=torch.float), y.to(device)
            out = model(x1, x2)
            loss = criterion(out, y)
            running_loss += loss.item()
            
            y_pred = torch.round(torch.sigmoid(out))
            correct = (y_pred == y).sum().float()
            acc = correct/y.shape[0]
            acc = torch.round(acc * 100)
            epoch_test_acc += acc
            
        epoch_test_loss = running_loss / len(test_loader)
        epoch_test_acc = epoch_test_acc / len(test_loader)

    return epoch_test_loss, epoch_test_acc


def main():
    
    if args.resume:
        checkpoint = args.resume
        model.load_state_dict(torch.load(checkpoint))
        resume = True
    else:
        model = Augmenting_Model(in_features=1, encoder_h=32)
        resume = False

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
        
    train_dataset = Normal_Dataset(raw_data_files = args.train_data) 
    test_dataset = Normal_Dataset(raw_data_files = args.test_data)  
    
    train_loader = data.DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=args.number_workers)
    test_loader = data.DataLoader(test_dataset, collate_fn=test_dataset.collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=args.number_workers)
    
    
    train_losses, test_losses, train_accs, test_accs, model = train_aug_ssl(train_loader, test_loader, model, n_epochs=args.epochs, lr=args.learning_rate, resume=resume)
    
    print(f'Best test accuracy {max(test_accs):.4f}')
    
    writer.close()

if __name__ == '__main__':
    main()