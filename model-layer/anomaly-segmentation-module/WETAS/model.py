import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()

        self.dilation = dilation

        self.diconv = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                dilation=dilation)

        self.conv1by1_skip = nn.Conv1d(in_channels=output_size,
                                       out_channels=output_size,
                                       kernel_size=1,
                                       dilation=1)

        self.conv1by1_out = nn.Conv1d(in_channels=output_size,
                                      out_channels=output_size,
                                      kernel_size=1,
                                      dilation=1)

    def forward(self, x):
        x = f.pad(x, (self.dilation, 0), "constant", 0)
        z = self.diconv(x)
        z = torch.tanh(z) * torch.sigmoid(z)
        s = self.conv1by1_skip(z)
        z = self.conv1by1_out(z) + x[:,:,-z.shape[2]:]
        return z, s

class DilatedCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, n_layers, pooling_type='max', granularity=1, \
      local_threshold=0.5, global_threshold=0.5, beta=10, split_size=500, dtw=None):
        super(DilatedCNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        
        self.rf_size = self.kernel_size ** self.n_layers # the size of receptive field
        self.pooling_type = pooling_type

        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.granularity = granularity
        self.beta = beta
        
        self.split_size = split_size
        self.dtw = dtw
        
        self.build_model(input_size, hidden_size, output_size, kernel_size, n_layers)

    def build_model(self, input_size, hidden_size, output_size, kernel_size, n_layers):
        # causal conv. layer
        self.causal_conv = nn.Conv1d(in_channels=input_size,
                                     out_channels=hidden_size,
                                     kernel_size=kernel_size,
                                     stride=1, dilation=1)

        # dilated conv. layer
        self.diconv_layers = nn.ModuleList()
        for i in range(n_layers):
            diconv = ResidualBlock(input_size=hidden_size,
                                   output_size=hidden_size,
                                   kernel_size=kernel_size,
                                   dilation=kernel_size**i)
            self.diconv_layers.append(diconv)

        # 1x1 conv. layer (for skip-connection)
        self.conv1by1_skip1 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=1, dilation=1)

        self.conv1by1_skip2 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=output_size,
                                        kernel_size=1, dilation=1)

        self.fc = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, x):
        x = x.transpose(1, 2)

        padding_size = self.rf_size - x.shape[2]
        if padding_size > 0:
            x = f.pad(x, (padding_size, 0), "constant", 0)
        
        x = f.pad(x, (1, 0), "constant", 0)
        z = self.causal_conv(x)

        out = torch.zeros(z.shape).cuda()
        for diconv in self.diconv_layers:
            z, s = diconv(z)
            out += s

        out = f.relu(out)
        out = self.conv1by1_skip1(out)
        out = f.relu(out)
        out = self.conv1by1_skip2(out).transpose(1, 2)
        return out

    def get_scores(self, x):
        ret = {}
        out = self.forward(x)
        ret['output'] = out

        # Compute weak scores
        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1) 
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]
        
        ret['wscore'] = torch.sigmoid(self.fc(_out).squeeze(dim=1))
        ret['wpred'] = (ret['wscore'] >= self.global_threshold).type(torch.cuda.FloatTensor)

        # Compute dense scores
        h = self.fc(out).squeeze(dim=2)
        ret['dscore'] = torch.sigmoid(h)
        ret['dpred'] = (ret['dscore'] >= self.local_threshold).type(torch.cuda.FloatTensor)
        return ret

    def get_seqlabel(self, actmap, wlabel):
        actmap *= wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1])
        seqlabel = (actmap >= self.local_threshold).type(torch.cuda.FloatTensor)
        seqlabel = f.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0)
        seqlabel = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity)))
        seqlabel = torch.max(seqlabel, dim=2)[0]

        seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).cuda(), seqlabel, torch.zeros(seqlabel.shape[0], 1).cuda()], dim=1)

        return seqlabel

    def dtw_loss(self, out, wlabel):
        h = self.fc(out).squeeze(dim=2) 
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2))

        with torch.no_grad():
            # Activation map
            actmap = h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            # Sequential labels
            pos_seqlabel = self.get_seqlabel(actmap, wlabel)
            neg_seqlabel = self.get_seqlabel(actmap, 1-wlabel)

        pos_dist = self.dtw(pos_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size
        neg_dist = self.dtw(neg_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size
        loss = f.relu(self.beta + pos_dist - neg_dist)
        return loss

    def get_alignment(self, label, score):
        # label : batch x pseudo-label length (= B x L)
        # score : batch x time-sereis length (= B x T)
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2  
        assert len(score.shape) == 2

        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1))
        indices = torch.max(A, dim=1)[1]
        return torch.gather(label, 1, indices)

    def get_dpred(self, out, wlabel):
        h = self.fc(out).squeeze(dim=2)
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2))

        with torch.no_grad():
            # Activation map
            actmap = h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            # Sequential labels
            seqlabel = self.get_seqlabel(actmap, wlabel)

        return self.get_alignment(seqlabel, dscore)

