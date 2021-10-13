import torch
from torch import optim
from torch.utils import data
from torch import nn

import numpy as np
from train_helpers import normalize, get_loss_weights, load_losses, save_losses
from downstream.TUH_edf import TUH_Normal_Abnormal
from sklearn.model_selection import KFold

from models import Supervised_TUH
#from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os.path as op
import os

import argparse

#from torch.utils.tensorboard import SummaryWriter

#Tensorboard    
#writer = SummaryWriter()

root = op.dirname(__file__)
saved_models_dir = op.join(root, 'saved_models')
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

parser = argparse.ArgumentParser(description='Downstream fine-tuning on different datasets.')
parser.add_argument('--batch-size', '-b', type=int, default=32, metavar='N', help='input batch size for fine-tuning (default: 32)')
parser.add_argument('--number-workers', '-n', type=int, default=16, metavar='N', help='number of works to load dataset (default: 16)')
parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--KFold', '-k', type=int, default=5, metavar='K', help='number of KFold (default: 5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
parser.add_argument('--learning-rate', type=float, default=5e-5, metavar='L', help='initial learning rate (default: 1e-4)')
parser.add_argument('--weight-decay', type=float, default=0., metavar='W', help='weight decay (default: 0)')

parser.add_argument('--load-model', type=str, default='', help='path to pre-trained model (default: None)')
parser.add_argument('--load-losses', type=str, default='', help='path to load latest checkpoint losses (default: None)')
parser.add_argument('--save-losses', type=str, default='', help='path to save new checkpoint losses (default: None)')
parser.add_argument('--save-name', type=str, default='checkpoint', help='path to save the final model')

parser.add_argument('--layers', type=int, default=4, help='number of layers to load')



args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set the random seed manually for reproducibility
torch.manual_seed(args.seed)

def train_test_loop(train_loader, test_loader, model, n_epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    #Run the training loop for defined number of epochs
    train_losses = []
    test_losses = []
    for epoch in range(1, n_epochs+1):
        train_loss, train_acc = _train_loss(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        test_loss, test_acc = _eval_loss(model, test_loader, criterion)
        test_losses.append(test_loss)
        print(f'Epoch {epoch}, Train loss {train_loss:.4f} , Train Accuracy {train_acc:.4f}, Test loss {test_loss:.4f}, Test Accuracy {test_acc:.4f}')
#         writer.add_scalar('Downstream Loss/train', train_loss, epoch)
#         writer.add_scalar('Downstream Loss/test', test_loss, epoch)
#         writer.add_scalar('Downstream Accuracy/train', train_acc, epoch)
#         writer.add_scalar('Downstream Accuracy/test', test_acc, epoch)
        scheduler.step()
#         writer.flush()
            
    #Save trained model
#     torch.save(model.state_dict(), op.join(root, 'saved_models', args.save_name + '.pt'))
#     save_losses(train_losses, test_losses, saved_models_dir, args.save_losses)
    
    return test_acc
    
    
def _train_loss(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    
    for i, data in enumerate(tqdm(train_loader)):
        x, y = data
        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
        optimizer.zero_grad()
        out = model(x)
        
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
   
        predicted = out.data.to('cpu')  #predicted are all 1?  
#         print(f'before round predicted = {predicted}')
        predicted = predicted.reshape(-1).detach().numpy().round()
        target = y.to('cpu')
        target = target.reshape(-1).detach().numpy()
#         print(f'predicted = {predicted}')
#         print(f'target = {target}')
        total += len(target)
        correct += sum(p == t for p, t in zip(predicted, target))
#        print(f'correct = {sum(p == t for p, t in zip(predicted, target))}')

    epoch_train_loss = total_loss / len(train_loader)
#    print(f'correct = {correct}, total = {total}')
    train_acc = correct / total
    
    return epoch_train_loss, train_acc


def _eval_loss(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            x, y = data           
            x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.float)
            out = model(x)
            
            loss = criterion(out, y)
            running_loss += loss.item()
            
            predicted = out.data.to('cpu') 
#             print(f'before round predicted = {predicted}')
            predicted = predicted.reshape(-1).detach().numpy().round()
            y = y.to('cpu')
            y = y.reshape(-1).detach().numpy()
#             print(f'predicted = {predicted}')
#             print(f'y = {y}')
            total += len(y)
            correct += sum(p == t for p, t in zip(predicted, y))
#            print(f'correct = {sum(p == t for p, t in zip(predicted, y))}')

            
        epoch_test_loss = running_loss / len(test_loader)
        
#        print(f'correct = {correct}, total = {total}')
        test_acc = correct / total
        

    return epoch_test_loss, test_acc

def main():
    
    #Load Dataset
    dataset = TUH_Normal_Abnormal(normal_filename='./downstream/tuh_normal.txt', abnormal_filename ='./downstream/tuh_abnormal.txt')
        
    
#     #Load model
#     model = Supervised_SSP_EEGBCI()
    
#     if args.load_model:
#         #Remove last linear layer parameters
#         pretrained_dict = torch.load(args.load_model) 
#         pretrained_param_names = list(pretrained_dict.keys())
#         model_dict = model.state_dict()
#         model_param_names = list(model_dict.keys())
        
#         # :4 load the first conv layer 
#         for i, _ in enumerate(pretrained_param_names[:4]):
#             model_dict[model_param_names[i]] = pretrained_dict[pretrained_param_names[i]]
            
#         model.load_state_dict(model_dict)
        
# #         #Freeze the first few layers
# #         i = 0
# #         for name, param in model.named_parameters():
# #             i += 1
# #             if param.requires_grad and i < 12:
# #                  param.requires_grad = False
#     model.to(device)
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
        
    
    folds = args.KFold
    kfold = KFold(n_splits=folds, shuffle=True)
    
    #For fold results
    results = {}
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold = {fold + 1}')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler)
        
        #################################################################
        #Load model
        model = Supervised_TUH()

        if args.load_model:
            pretrained_dict = torch.load(args.load_model) 
            pretrained_param_names = list(pretrained_dict.keys())
            model_dict = model.state_dict()
            model_param_names = list(model_dict.keys())

            # :4 load the first conv layer  :8 load the first two conv layers :12 thrid :16 forth :20 fifth :24 sixth 
            for i, _ in enumerate(pretrained_param_names[:args.layers]):
                model_dict[model_param_names[i]] = pretrained_dict[pretrained_param_names[i]]

            model.load_state_dict(model_dict)
            
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        ##################################################################
        
        test_acc = train_test_loop(train_loader, test_loader, model, n_epochs=args.epochs, lr=args.learning_rate)
        results[fold] = test_acc
        
    f = open("tuh_downstream_results.txt", "a")
    f.write(f"initial layers {args.layers}\n")
    # Print fold results
    f.write(f'K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS\n')
    f.write('--------------------------------\n')
    sum = 0.0
    for key, value in results.items():
        f.write(f'Fold {key}: {value} %\n')
        sum += value
    f.write(f'Average: {sum/len(results.items())} %\n')
    f.close()
    print(f'tuh initial layers {args.layers} done!')
#     writer.close()
    

if __name__ == '__main__':
    main()