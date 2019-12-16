import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir, batch_size, valid_size=0.1, num_workers=4, pin_memory=False):

	# MNIST data description: mean = 0.1307, std = 0.3081
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load MNIST dataset
    dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_data = len(dataset)
    indices = list(range(num_data))
    split = int(np.floor(valid_size * num_data))
    np.random.shuffle(indices)

    # split dataset => training : validation = 9 : 1 
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Load training, validation data from MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):

	# MNIST data description: mean = 0.1307, std = 0.3081
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load MNIST dataset
    dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    # Load test data from MNIST dataset
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader



# DNN network model for MNIST dataset
class MnistNetwork(nn.Module):
    def __init__(self, conv1out, conv2out, fc1in, fc2in):
        super(MnistNetwork, self).__init__()
        # 2 convolutional layer + 2 fully connected layer
        self.con1out = conv1out
        self.con2out = conv2out
        self.fc1in = fc1in
        self.fc2in = fc2in
        self.conv1 = nn.Conv2d(1, conv1out, 5, 1)
        self.conv2 = nn.Conv2d(conv1out, conv2out, 5, 1)
        self.fc1 = nn.Linear(fc1in, fc2in)
        self.fc2 = nn.Linear(fc2in, 10)

    def forward(self, x):
    	# activation function: relu
    	# pooling: max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.fc1in)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# Train the model
def train(model, device, train_loader, optimizer, epoch, num_data):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= num_data

    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))

# Evaluate the test(or validation) performance 
def test(model, device, test_loader, num_data, set_type):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_data

    print '\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
        set_type, test_loss, correct, num_data,
        100. * correct / num_data)

    return test_loss, 100. * correct / num_data

if __name__ == '__main__':
    
    ## Learning configuration
	learning_rate = 0.01
	batch_size = 64
	valid_size = 0.1
	epochs = 50

	## Model configuation
	conv1out = 20 # output size of the 1st conv layer
	conv2out = 50 # output size of the 2nd conv layer
	fc1in = 4*4*50 # input size of the 1st fully connected layer
	fc2in = 500 # input size of the 2nd fully connected layer

    # To use GPU if possible
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

    # Load train, validation dataset
	train_loader, valid_loader = get_train_valid_loader('data', batch_size=batch_size, valid_size=valid_size, num_workers=4, pin_memory=False)

    # Load test set
	test_loader = get_test_loader('data', batch_size=batch_size)

    # Build a DNN model
	model = MnistNetwork(conv1out, conv2out, fc1in, fc2in).to(device)

    # Optimizer: SGD
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    # Start training
	best_val_loss = 100
	best_val_acc = 0
	best_test_loss = 100
	best_test_acc = 0
	best_loss_epoch = 0
	best_acc_epoch = 0

	print "Training..."
	for epoch in range(0, epochs):
		print "Epoch " + str(epoch + 1)

		# Train model 
		train(model, device, train_loader, optimizer, epoch, int(float(60000) * (1.0 - valid_size)))

		# Evaluate validation dataset
		val_loss, val_acc = test(model, device, valid_loader, int(float(60000) * valid_size), 'Valid')

		# Evaluate test dataset
		test_loss, test_acc = test(model, device, test_loader, 10000, 'Test')

		# Report the test and validation performance when the validation performance is the best.
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_test_loss = test_loss
			best_loss_epoch = epoch
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_test_acc = test_acc
			best_acc_epoch = epoch

	# Print results
	print "Train Finished.\n\n"
	print "Valid loss: " + str(best_val_loss) + " at epoch " + str(best_loss_epoch)
	print "Valid acc : " + str(best_val_acc) + '%' + " at epoch " + str(best_acc_epoch)

	print "Test loss: " + str(best_test_loss)
	print "Test acc : " + str(best_test_acc) + '%'
