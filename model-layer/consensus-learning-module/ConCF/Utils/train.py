import time
import torch
from Utils.utils import *


def train_singleCF(model, gpu, optimizer, train_dict, test_dict):

	train_loader = train_dict['train_loader']
	test_dataset = test_dict['test_dataset']

	train_loss_arr = []
	train_history_R = []
	train_history_N = []
	p = 10

	for ep in range(train_dict['max_epoch']):

		ts1 = time.time()
		train_loader.dataset.ns()
		ep_loss = []

		for mini_b in train_loader:

			tmp = {key: value.to(gpu) for key, value in mini_b.items()}
			mini_b = train_loader.dataset.for_AE_b(tmp, gpu)

			model.train()
			output = model(mini_b)

			b_loss = model.get_loss(output)

			optimizer.zero_grad()
			b_loss.backward()
			optimizer.step()

			ep_loss.append(b_loss.data)

		ep_loss = torch.mean(torch.stack(ep_loss)).data.cpu().numpy()
		train_loss_arr.append(ep_loss)

		te1 = time.time()
		
		if ep % p == 0:			
			model.eval()
			with torch.no_grad():
				ts2 = time.time()
				eval_results = evaluate_singleCF(model, gpu, train_loader, test_dict)
				train_history_R.append(eval_results['valid']['R50'])
				train_history_N.append(eval_results['valid']['N50'])
				te2 = time.time()

	return train_history_R, train_history_N



def train(model, gpu, optimizer, train_dict, test_dict):

	weight_params, Queue_R, S = setup(gpu)
	weight_optimizer = torch.optim.Adam(weight_params, lr=train_dict['lr'])
	L1_loss = nn.L1Loss()

	train_loader = train_dict['train_loader']
	test_dataset = test_dict['test_dataset']

	train_loss_arr = []
	train_history_R = []
	train_history_N = []

	C_history_R = []
	C_history_N = []

	con_top_results = None
	con_results = None
	is_CL_begin = False
	cl_alpha = train_dict['alpha']
	Queue_size = 5
	p = 10

	for ep in range(train_dict['max_epoch']):

		ts1 = time.time()
		train_loader.dataset.ns()
		ep_loss = []

		for mini_b in train_loader:

			tmp = {key: value.to(gpu) for key, value in mini_b.items()}
			mini_b = train_loader.dataset.for_AE_b(tmp, gpu)

			model.train()
			output = model(mini_b)

			l1, l2, l3, l4, l5, lb = model.get_loss(output, weight_params)
			b_loss = l1 + l2 + l3 + l4 + l5 + lb

			## For the first few eps
			if ep <= 10:
				l1_0, l2_0, l3_0, l4_0, l5_0, lb_0 = l1.data, l2.data, l3.data, l4.data, l5.data, lb.data

			if con_top_results != None:
				b_user = mini_b['user']
				KD_item = torch.index_select(con_top_results, 0, b_user.unique())

				cl1, cl2, cl3, cl4, cl5 = model.CL_loss(b_user, KD_item, weight_params)
				cl_loss = cl1 + cl2 + cl3 + cl4 + cl5
				b_loss += cl_loss * cl_alpha

			optimizer.zero_grad()
			b_loss.backward(retain_graph=True)
			
			# For balancing
			if con_top_results == None:
				G1 = torch.norm(torch.autograd.grad(l1, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l1, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G2 = torch.norm(torch.autograd.grad(l2, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l2, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G3 = torch.norm(torch.autograd.grad(l3, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l3, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G4 = torch.norm(torch.autograd.grad(l4, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l4, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G5 = torch.norm(torch.autograd.grad(l5, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) 
			else:
				G1 = torch.norm(torch.autograd.grad(l1 + cl1 * cl_alpha, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l1 + cl1 * cl_alpha, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G2 = torch.norm(torch.autograd.grad(l2 + cl2 * cl_alpha, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l2 + cl2 * cl_alpha, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G3 = torch.norm(torch.autograd.grad(l3 + cl3 * cl_alpha, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l3 + cl3 * cl_alpha, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G4 = torch.norm(torch.autograd.grad(l4 + cl4 * cl_alpha, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) + torch.norm(torch.autograd.grad(l4 + cl4 * cl_alpha, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
				G5 = torch.norm(torch.autograd.grad(l5 + cl5 * cl_alpha, model.user_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) 

			Gb = torch.norm(torch.autograd.grad(lb, model.item_emb.parameters(), retain_graph=True, create_graph=True)[0], 2) / 2.
			G_avg = (G1 + G2 + G3 + G4 + G5 + Gb) / 6
			lhat1, lhat2, lhat3, lhat4, lhat5, lhatb = l1/l1_0, l2/l2_0, l3/l3_0, l4/l4_0, l5/l5_0, lb/lb_0
			lhat_avg = (lhat1 + lhat2 + lhat3 + lhat4 + lhat5 + lhatb) / 6

			tau = 0.15
			inv_rate1, inv_rate2, inv_rate3, inv_rate4, inv_rate5, inv_rateb = lhat1/lhat_avg, lhat2/lhat_avg, lhat3/lhat_avg, lhat4/lhat_avg, lhat5/lhat_avg, lhatb/lhat_avg
			C1, C2, C3, C4, C5, Cb = G_avg * (inv_rate1 ** tau), G_avg * (inv_rate2 ** tau), G_avg * (inv_rate3 ** tau), G_avg * (inv_rate4 ** tau), G_avg * (inv_rate5 ** tau),  G_avg * (inv_rateb ** tau)
			C1, C2, C3, C4, C5, Cb = C1[0].detach(), C2[0].detach(), C3[0].detach(), C4[0].detach(), C5[0].detach(), Cb[0].detach()

			weight_optimizer.zero_grad()
			weight_loss = L1_loss(G1, C1) + L1_loss(G2, C2) + L1_loss(G3, C3) + L1_loss(G4, C4) + L1_loss(G5, C5) + L1_loss(Gb, Cb)

			weight_loss.backward()
			
			weight_optimizer.step()
			optimizer.step()

			# renormalize
			coef = 6 / torch.stack(weight_params).sum()
			for i in range(6):
				weight_params[i].data = torch.clamp(weight_params[i].data * coef, min=0.01)

			ep_loss.append(b_loss.data)

		ep_loss = torch.mean(torch.stack(ep_loss)).data.cpu().numpy()
		train_loss_arr.append(ep_loss)

		te1 = time.time()
		
		if ep % p == 0:			
			model.eval()
			with torch.no_grad():
				ts2 = time.time()
				eval_results, rank_mats = evaluate(model, gpu, train_loader, test_dict)
				te2 = time.time()

				# warm-up
				if not is_CL_begin:
					eval_best = max([eval_results[i]['valid']['R50'] for i in range(5)])

					for i in range(len(eval_results)):
						Queue_R[i].append(rank_mats[i])

						if len(Queue_R[i]) == Queue_size:
							S.append(np.std(np.stack(Queue_R[i], axis=-1), axis=-1))
							is_CL_begin = True
				else:
					con_score_mat = con_gen(Queue_R, S=S)
					con_sorted_mat = np.argsort(-con_score_mat, axis=-1)
					con_results = get_eval_np(train_loader.dataset.rating_mat, test_dataset.valid_mat, test_dataset.test_mat, con_sorted_mat)
					
					eval_best = con_results['valid']['R50']

					con_top_results = con_sorted_mat[:, :100]						# for fast computation
					con_top_results = torch.LongTensor(con_top_results).to(gpu)

					if (ep <= p * 5):

						for i in range(len(eval_results)):
							Queue_R[i].append(rank_mats[i])
							if len(Queue_R[i]) > Queue_size:
								Queue_R[i].pop(0)

							S[i] = np.std(np.stack(Queue_R[i], axis=-1), axis=-1)
					else:
						for i in range(len(eval_results)):
							Queue_R[i].append(rank_mats[i])
							if len(Queue_R[i]) > Queue_size:
								Queue_R[i].pop(0)

				train_history_R.append(eval_results[0]['valid']['R50'])
				train_history_N.append(eval_results[0]['valid']['N50'])

				if con_results == None:
					C_history_R.append(eval_results[0]['valid']['R50'])
					C_history_N.append(eval_results[0]['valid']['N50'])
				else:
					C_history_R.append(con_results['valid']['R50'])
					C_history_N.append(con_results['valid']['N50'])

	return train_history_R, C_history_R, train_history_N, C_history_N