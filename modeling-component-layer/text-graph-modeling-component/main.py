import argparse
from data import AMT
from model import GNN
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, required=True)
args = parser.parse_args()

embed_size = 64
dropout = 0.5
learning_rate = 0.01
num_epochs = 100

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

data_train, data_test = AMT(device)

model = GNN(*data_train.size, embed_size, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    y_pred = model(data_train).reshape(-1)
    y = data_train.edge_type.float()

    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    rmse_loss_train = loss.item() ** 0.5

    model.eval()
    with torch.no_grad():
        y_pred = model(data_test).reshape(-1)
        y = data_test.edge_type.float()

        rmse_loss_test = criterion(y_pred, y).item() ** 0.5

    print(f' [Epoch {epoch + 1:3}/{num_epochs}]', end='  ')
    print(f'RMSE Loss (Train: {rmse_loss_train:6.4f} | Test: {rmse_loss_test:6.4f})')

torch.save(model.state_dict(), f'model/model.ckpt')
