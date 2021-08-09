import argparse
import random
import torch
import torch.optim as optim
import metric
from model import ProxySR
import pandas as pd 
from dataset import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=512)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--k', type=int, default=300)
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--margin', type=float, default=0.1)
parser.add_argument('--lambda_dist', type=float, default=0.2)
parser.add_argument('--lambda_orthog', type=float, default=0.0)
parser.add_argument('--E', type=int, default=10)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_position', type=int, default=50)
parser.add_argument('--t0', type=float, default=3.0)
parser.add_argument('--te', type=float, default=0.01)
parser.add_argument('--num_epoch', type=int, default=500)
parser.add_argument('--repetitive', type=bool, default=False)
args = parser.parse_args()

dataset = args.dataset
batch_size = args.batch_size
val_batch_size = args.val_batch_size
embed_dim = args.embed_dim
lr = args.lr
k = args.k
num_epoch = args.num_epoch
dropout_rate = args.dropout_rate
margin = args.margin
max_position = args.max_position
lambda_dist = args.lambda_dist
lambda_orthog = args.lambda_orthog
E = args.E
patience = args.patience
t0 = args.t0
te = args.te
repetitive = args.repetitive

K = [5, 10, 20]


def main():
    print('Loading data...')
    
    train_data = pd.read_csv(dataset + "/train.csv").fillna('NA')
    val_data = pd.read_csv(dataset + "/valid.csv").fillna('NA')
    test_data = pd.read_csv(dataset + "/test.csv").fillna('NA')

    num_items = train_data['itemId'].max() + 1
    
    train_iid, train_length, train_label = preprocess(train_data, repetitive, train=True)
    val_iid, val_length, val_label = preprocess(val_data, repetitive, train=False)
    test_iid, test_length, test_label = preprocess(test_data, repetitive, train=False)

    train_data = zip(train_iid, train_length, train_label)
    val_data = zip(val_iid, val_length, val_label)
    test_data = zip(test_iid, test_length, test_label)

    model = ProxySR(num_items, embed_dim, k, max_position, dropout_rate, margin).to(device)

    optimizer = optim.Adam(model.parameters(), lr)

    best_recall20 = 0
    best_epoch = 0
    
    for epoch in range(0, num_epoch):
        print('')
        print "Epoch: " + str(epoch)

        random.shuffle(train_data)

        # temperature annealing
        tau = max(t0 * ((te / t0) ** (float(epoch) / E)), te)

        train_model(train_data, model, optimizer, batch_size, tau)

        recalls, mrrs = validate(val_data, model, K, tau)

        print('')
        print('Validation data:')
        print('Recall@5 : ' + str(round(recalls[0], 4)) + ', MRR@5 : ' + str(round(mrrs[0], 4)))
        print('Recall@10: ' + str(round(recalls[1], 4)) + ', MRR@10: ' + str(round(mrrs[1], 4)))
        print('Recall@20: ' + str(round(recalls[2], 4)) + ', MRR@20: ' + str(round(mrrs[2], 4)))
        
        if recalls[2] > best_recall20:
            best_recall20 = recalls[2]
            best_epoch = epoch
            best_recalls = list(recalls)
            best_mrrs = list(mrrs)

            test_recalls, test_mrrs = validate(test_data, model, K, tau)

            print('')
            print('Best Recall@5 : ' + str(round(best_recalls[0], 4)) + ', Best MRR@5 : ' + str(round(best_mrrs[0], 4)))
            print('Best Recall@10: ' + str(round(best_recalls[1], 4)) + ', Best MRR@10: ' + str(round(best_mrrs[1], 4)))
            print('Best Recall@20: ' + str(round(best_recalls[2], 4)) + ', Best MRR@20: ' + str(round(best_mrrs[2], 4)))

            print('')
            print('### Test Recall@5 : ' + str(round(test_recalls[0], 4)) + ', Test MRR@5 : ' + str(round(test_mrrs[0], 4)))
            print('### Test Recall@10: ' + str(round(test_recalls[1], 4)) + ', Test MRR@10: ' + str(round(test_mrrs[1], 4)))
            print('### Test Recall@20: ' + str(round(test_recalls[2], 4)) + ', Test MRR@20: ' + str(round(test_mrrs[2], 4)))

        if best_epoch + patience < epoch:
            break


def pad(l, limit, p):
    max_len = limit
    l = list(map(lambda x: [p] * (max_len - min(len(x), limit)) + x[:min(len(x), limit)], l))
    return l


def train_model(train_data, model, optimizer, batch_size, tau):
    model.train()

    sum_epoch_target_loss = 0
    for i in range(0, len(train_data), batch_size):
        if i + batch_size >= len(train_data):
            train_batch = train_data[i:]
        else:
            train_batch = train_data[i: i + batch_size]

        sess, length, target = zip(*train_batch)

        max_length = int(max(length))
        sess = torch.tensor(pad(sess, max_length, 0)).to(device)
        target = torch.tensor(pad(target, max_length, 0)).to(device)
        length = torch.tensor(length).to(device)
        
        optimizer.zero_grad()

        target_flatten = target.view(-1) 
        sess_cum = sess.repeat(1, max_length).view(-1, max_length) * torch.tril(torch.ones((max_length, max_length), device=device), diagonal=0).repeat(sess.shape[0], 1).to(torch.long)
        length3 = (torch.sum(sess_cum != 0, dim=1) >= 3).to(device)
        if repetitive:
            valid = length3
        else:
            unseen_item = torch.sum((sess_cum - target_flatten.view(-1, 1)).eq(0), dim=1).eq(0).to(device)
            valid = (length3 & unseen_item)
        valid_ind = valid.nonzero().squeeze()

        valid_target = target_flatten[valid_ind]
        distance, orthogonal = model(sess, length, tau, valid, train=True)
            
        target_distance = distance[range(len(distance)), valid_target]
        margin_distance = torch.max(margin + target_distance.unsqueeze(1) - distance, torch.zeros_like(distance).to(device))
        margin_distance[:, 0] = 0
        margin_distance, _ = torch.topk(margin_distance, k=100, dim=-1)
        target_loss = torch.mean(margin_distance)

        reg_dist = torch.mean(target_distance)
        reg_orthog = torch.mean(orthogonal)

        # Loss
        loss = target_loss + lambda_dist * reg_dist + lambda_orthog * reg_orthog

        loss.backward()
        optimizer.step() 

        loss_target = target_loss.item()
        
        sum_epoch_target_loss += loss_target
        
        if (i / batch_size) % (len(train_data)/batch_size / 5) == (len(train_data)/batch_size / 5) - 1:
            print 'Loss: ' + str(round(loss_target, 4)) + ' (avg: ' + str(round(sum_epoch_target_loss / (i/batch_size + 1), 4)) + ')'
        
def validate(val_data, model, K, tau):
    model.eval()

    recalls = [0.0, 0.0, 0.0]
    mrrs = [0.0, 0.0, 0.0]

    len_val = 0

    with torch.no_grad():
        for i in range(0, len(val_data), val_batch_size):
            if i + val_batch_size >= len(val_data):
                val_batch = val_data[i:]
            else:
                val_batch = val_data[i: i + val_batch_size]

            sess, length, target = zip(*val_batch)

            max_length = int(max(length))
            sess = torch.tensor(pad(sess, max_length, 0)).to(device)
            target = torch.tensor(target).to(device)
            length = torch.tensor(length).to(device)
            
            length3 = (torch.sum(sess != 0, dim=1) >= 3).to(device)
            if repetitive:
                valid = length3
            else:
                unseen_item = torch.sum((sess - target.view(-1, 1)).eq(0), dim=1).eq(0).to(device)
                valid = (length3 & unseen_item)
            target *= torch.tensor(valid).to(torch.long).to(device)
            distance = model(sess, length, tau, valid, train=False)

            if not repetitive:
                distance[range(len(distance)), sess.t()] = float('inf')
            distance[:, 0] = float('inf')
            len_val += torch.sum(valid).item()
            valid_ind = valid.nonzero().squeeze()
            target = target[valid_ind]

            recall, mrr = metric.evaluate(distance, target, K=K)

            recalls[0] += recall[0]
            mrrs[0] += mrr[0]
            recalls[1] += recall[1]
            mrrs[1] += mrr[1]
            recalls[2] += recall[2]
            mrrs[2] += mrr[2]

    recalls = list(map(lambda x: x / len_val, recalls))
    mrrs = list(map(lambda x: x / len_val, mrrs))

    return recalls, mrrs


if __name__ == '__main__':
    main()
