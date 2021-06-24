import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# MLP with a Softmax classifier

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes[:-1]
        self.latent_size = hidden_sizes[-1]
        self.num_classes = num_classes

        self.build_fe()
        self.init_fe_weights()

    def build_fe(self):
        layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes
       
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], self.latent_size))
        layers.append(nn.Linear(self.latent_size, self.num_classes))
        self.layers = nn.ModuleList(layers)

    def init_fe_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
        return x

    def forward(self, x):
        out = self._forward(x)
        score = self.layers[-1](out)
        return score

    def feature_list(self, x):
        out = self._forward(x)
        score = self.layers[-1](out)
        return score, [out]

    def intermediate_forward(self, x, layer_index):
        out = self._forward(x)
        return out


# MLP with a Deep-MCDD classifier  

class MLP_DeepMCDD(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP_DeepMCDD, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes[:-1]
        self.latent_size = hidden_sizes[-1]
        self.num_classes = num_classes

        self.centers = torch.nn.Parameter(torch.zeros([num_classes, self.latent_size]), requires_grad=True)
        self.alphas = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        self.logsigmas = torch.nn.Parameter(torch.zeros(num_classes), requires_grad=True)
        
        self.build_fe()
        self.init_fe_weights()

    def build_fe(self):
        layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes
       
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], self.latent_size))
        self.layers = nn.ModuleList(layers)

    def init_fe_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.centers)
        nn.init.zeros_(self.alphas)
        nn.init.zeros_(self.logsigmas)

    def _forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def forward(self, x):
        out = self._forward(x)
        out = out.unsqueeze(dim=1).repeat([1, self.num_classes, 1])
        scores = torch.sum((out - self.centers)**2, dim=2) / 2 / torch.exp(2 * F.relu(self.logsigmas)) + self.latent_size * F.relu(self.logsigmas)
        return scores


# MLP with a Soft-MCDD classifier  

class MLP_SoftMCDD(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, epsilon=0.1):
        super(MLP_SoftMCDD, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes[:-1]
        self.latent_size = hidden_sizes[-1]
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.centers = torch.zeros([num_classes, self.latent_size]).cuda()
        self.radii = torch.ones(num_classes).cuda()

        self.build_fe()
        self.init_fe_weights()

    def build_fe(self):
        layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes
       
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], self.latent_size))
        self.layers = nn.ModuleList(layers)

    def init_fe_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.centers)

    def init_centers(self):
        nn.init.xavier_uniform_(self.centers)
        self.centers = 10 * self.centers / torch.norm(self.centers, dim=1)

    def update_centers(self, data_loader):
        class_outputs = {i : [] for i in range(self.num_classes)}
        centers = torch.zeros([self.num_classes, self.latent_size]).cuda()

        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self._forward(inputs)

                for k in range(self.num_classes):
                    indices = (labels == k).nonzero().squeeze(dim=1)
                    class_outputs[k].append(outputs[indices])

        for k in range(self.num_classes):
            class_outputs[k] = torch.cat(class_outputs[k], dim=0)
            centers[k] = torch.mean(class_outputs[k], dim=0)

        self.centers.data = centers

    def update_radii(self, data_loader):
        class_scores = {i : [] for i in range(self.num_classes)}
        radii = np.zeros(self.num_classes)

        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                scores = self.forward(inputs)

                for k in range(self.num_classes):
                    indices = (labels == k).nonzero().squeeze(dim=1)
                    class_scores[k].append(torch.sqrt(scores[indices][:, k]))

        for k in range(self.num_classes):
            class_scores[k] = torch.cat(class_scores[k], dim=0)
            radii[k] = np.quantile(class_scores[k].cpu().numpy(), 1 - self.epsilon)

        self.radii = torch.Tensor(radii).cuda()

    def update_centers_and_radii(self, data_loader):
        class_outputs = {i : [] for i in range(self.num_classes)}
        class_scores = {i : [] for i in range(self.num_classes)}
        centers = torch.zeros([self.num_classes, self.latent_size]).cuda()
        radii = np.zeros(self.num_classes)

        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs, scores = self._forward(inputs), self.forward(inputs)

                for k in range(self.num_classes):
                    indices = (labels == k).nonzero().squeeze(dim=1)
                    class_outputs[k].append(outputs[indices])
                    class_scores[k].append(torch.sqrt(scores[indices][:, k]))

        for k in range(self.num_classes):
            class_outputs[k] = torch.cat(class_outputs[k], dim=0)
            class_scores[k] = torch.cat(class_scores[k], dim=0)
            centers[k] = torch.mean(class_outputs[k], dim=0)
            radii[k] = np.quantile(class_scores[k].cpu().numpy(), 1 - self.epsilon)

        self.centers.data = centers
        self.radii = torch.Tensor(radii).cuda()

    def _forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def forward(self, x):
        out = self._forward(x)
        out = out.unsqueeze(dim=1).repeat([1, self.num_classes, 1])
        scores = torch.sum((out - self.centers)**2, dim=2)
        return scores

