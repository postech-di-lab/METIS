import torch.nn.functional as F
import torch.nn as nn
import torch
from pdb import set_trace as bp
import numpy as np
from Utils.data_utils import count_parameters
import math
from Models.BPR import BPR


class Expert(nn.Module):
	def __init__(self, dims):
		super(Expert, self).__init__()

		self.mlp = nn.Sequential(nn.Linear(dims[0], dims[1]),
							nn.ReLU(),
							nn.Linear(dims[1], dims[2])
							)

	def forward(self, x):
		return self.mlp(x)


class BPR_DE(BPR):
	def __init__(self, user_count, item_count, user_emb_teacher, item_emb_teacher, gpu, student_dim, num_experts):

		BPR.__init__(self, user_count, item_count, student_dim, gpu)

		self.student_dim = student_dim
		self.gpu = gpu

		# Teacher Embedding
		self.user_emb_teacher = nn.Embedding.from_pretrained(user_emb_teacher)
		self.item_emb_teacher = nn.Embedding.from_pretrained(item_emb_teacher)

		self.user_emb_teacher.weight.requires_grad = False
		self.item_emb_teacher.weight.requires_grad = False

		self.teacher_dim = self.user_emb_teacher.weight.size(1)

		num_emb_params = self.user_emb.weight.size(0) * self.user_emb.weight.size(1) + self.item_emb.weight.size(0) * self.item_emb.weight.size(1)


		# Expert Configuration
		self.num_experts = num_experts
		expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim) // 2, self.teacher_dim]

		## for self-distillation
		if self.teacher_dim == self.student_dim:
			expert_dims = [self.student_dim, self.student_dim // 2, self.teacher_dim]

		self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
		self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

		self.user_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, num_experts), nn.Softmax(dim=1))
		self.item_selection_net = nn.Sequential(nn.Linear(self.teacher_dim, num_experts), nn.Softmax(dim=1))

		num_experts_params = count_parameters(self.user_experts) + count_parameters(self.item_experts)
		num_gates_params = count_parameters(self.user_selection_net) + count_parameters(self.item_selection_net)


		print("Teacher Dim::", self.teacher_dim, "Student Dim::", self.student_dim)
		print("Expert dims::", expert_dims)
		print("Num. embedding parameter::",  num_emb_params, " Num. gate parameter::", num_gates_params, " Num. expert parameter::", num_experts_params)
		print("Total parameters::", num_emb_params + num_gates_params + num_experts_params)

		self.T = 0.
		self.sm = nn.Softmax(dim = 1)	


	def get_DE_loss(self, batch_entity, is_user=True):

		if is_user:
			s = self.user_emb(batch_entity)													
			t = self.user_emb_teacher(batch_entity)										

			experts = self.user_experts
			selection_net = self.user_selection_net

		else:
			s = self.item_emb(batch_entity)													
			t = self.item_emb_teacher(batch_entity)											
	
			experts = self.item_experts
			selection_net = self.item_selection_net

		selection_dist = selection_net(t) 								# batch_size x num_experts

		if self.num_experts == 1:
			selection_result = 1.
		else:
			# Expert Selection
			g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(self.gpu)
			eps = 1e-10 										# for numerical stability
			selection_dist = selection_dist + eps
			selection_dist = self.sm((selection_dist.log() + g) / self.T)

			selection_dist = torch.unsqueeze(selection_dist, 1)					# batch_size x 1 x num_experts
			selection_result = selection_dist.repeat(1, self.teacher_dim, 1)			# batch_size x teacher_dims x num_experts

		expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)] 		# s -> t
		expert_outputs = torch.cat(expert_outputs, -1)							# batch_size x teacher_dims x num_experts

		expert_outputs = expert_outputs * selection_result						# batch_size x teacher_dims x num_experts
		expert_outputs = expert_outputs.sum(2)								# batch_size x teacher_dims	

		DE_loss = ((t-expert_outputs) ** 2).sum(-1).sum() 

		return DE_loss 
