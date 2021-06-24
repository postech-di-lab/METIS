import os, argparse
import torch
import numpy as np
from torchvision import transforms

import models
from dataloader_image import get_id_image_data, get_ood_image_data
from utils import compute_confscores, compute_metrics, print_ood_results

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='svhn | cifar10 | cifar100')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--datadir', default='./image_data/', help='path to dataset')
parser.add_argument('--outdir', default='./output/', help='folder to output results')
parser.add_argument('--modeldir', default='./trained_model/', help='folder to trained model')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loader')
parser.add_argument('--num_epochs', type=int, default=200, help='the number of epochs for training sc-layers')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate of SGD optimizer')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for nesterov momentum of SGD optimizer')
parser.add_argument('--reg_lambda', type=float, default=0.1, help='regularization coefficient')
parser.add_argument('--pretrained', type=bool, default=False, help='initialize the network with pretrained weights or random weights')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    outdir = os.path.join(args.outdir, args.net_type + '_' + args.dataset)
    pretrained_path = os.path.join('./pretrained/', args.net_type + '_' + args.dataset + '.pth')
    model_path = os.path.join(args.modeldir, args.net_type + '_' + args.dataset + '.pth')
    
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    if os.path.isdir(args.modeldir) == False:
        os.mkdir(args.modeldir)

    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(args.gpu)
    
    if args.dataset == 'svhn':
        num_classes = 10
        ood_list = ['cifar10', 'imagenet_crop', 'lsun_crop']
    elif args.dataset == 'cifar10':
        num_classes = 10
        ood_list = ['svhn', 'imagenet_crop', 'lsun_crop']
    elif args.dataset == 'cifar100':
        num_classes = 100
        ood_list = ['svhn', 'imagenet_crop', 'lsun_crop']
        
    if args.net_type == 'densenet':
        model = models.DenseNet_DeepMCDD(num_classes=num_classes)
        if args.pretrained == True:
            model.load_fe_weights(torch.load(pretrained_path, map_location = "cuda:" + str(args.gpu)))
        in_transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), 
                            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                            ])
        in_transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])

    elif args.net_type == 'resnet':
        model = models.ResNet_DeepMCDD(num_classes=num_classes)
        if args.pretrained == True:
            model.load_fe_weights(torch.load(pretrained_path, map_location = "cuda:" + str(args.gpu)))
        in_transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        in_transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
 
    model.cuda()       
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.num_epochs*0.5), int(args.num_epochs*0.75)], gamma=0.1)
    
    train_loader, test_id_loader = get_id_image_data(args.dataset, args.batch_size, in_transform_train, in_transform_test, args.datadir)
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            dists = model(images)
            scores = - dists + model.alphas

            label_mask = torch.zeros(labels.size(0), num_classes).cuda().scatter_(1, labels.unsqueeze(dim=1), 1)

            pull_loss = torch.mean(torch.sum(torch.mul(label_mask, dists), dim=1))
            push_loss = ce_loss(scores, labels)
            loss = args.reg_lambda * pull_loss + push_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
        
        scheduler.step()

        model.eval()
        with torch.no_grad():
            # (1) evaluate ID classification
            correct, total = 0, 0
            for images, labels in test_id_loader:
                images, labels = images.cuda(), labels.cuda()
                scores = - model(images) + model.alphas
                _, predicted = torch.max(scores, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            idacc = 100 * correct / total
                
            ood_results_list = []
            compute_confscores(model, test_id_loader, outdir, True)

            for ood in ood_list:
                test_ood_loader = get_ood_image_data(ood, args.batch_size, in_transform_test, args.datadir)
                compute_confscores(model, test_ood_loader, outdir, False)
                ood_results_list.append(compute_metrics(outdir))

        print('== Epoch [{}/{}], Loss {} =='.format(epoch+1, args.num_epochs, total_loss))   
        print('ID Accuracy on "{idset:s}" test images : {val:6.2f}\n'.format(idset=args.dataset, val=idacc))
        for ood_idx, ood_results in enumerate(ood_results_list):
            print('OOD accuracy on "{oodset:s}" test samples :'.format(oodset=ood_list[ood_idx]))
            print_ood_results(ood_results)

    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()
