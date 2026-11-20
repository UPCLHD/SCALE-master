# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
# Update: 01/08/2022
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class SCALE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SCALE, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        self.knn_k_uu = config['knn_k_uu']
        self.knn_k_ii = config['knn_k_ii']
        self.knn_k_ui = config['knn_k_ui']
        self.branch_cl_tau = config['branch_cl_tau']
        self.branch_cl_reg = config['branch_cl_reg']
        self.final_cl_tau = config['final_cl_tau']
        self.final_cl_reg = config['final_cl_reg']
        self.num_neg = config['num_neg']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        self.sc_user, self.sc_item, self.sc_ui = self.get_sc_mat_dataset_file(config['dataset'], self.knn_k_uu, self.knn_k_ii, self.knn_k_ui, config)
        # item_item
        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.mm_adj = image_adj
        if self.t_feat is not None:
            # indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            indices, text_adj = self.get_knn_adj_mat_ad_sc(self.text_embedding.weight.detach(), self.sc_item)
            self.mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj
            del image_adj
            # torch.save(self.mm_adj, mm_adj_file)
        # user_user
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        user_file_path = os.path.join(dataset_path, config['user_feature_file'])
        self.user_feat = torch.from_numpy(np.load(user_file_path, allow_pickle=True)).type(torch.FloatTensor).to( self.device)
        uu_indices, self.user_mm_adj = self.get_knn_adj_mat_ad_sc(self.user_feat.detach(), self.sc_user)
        uu_indices = uu_indices.cpu()
        # user_item
        ui_indices, _ = self.get_knn_adj_mat_user_item(self.user_feat.detach(), self.text_embedding.weight.detach(), self.knn_k)
        ui_indices = ui_indices.cpu()
        ui_indices = torch.cat((ui_indices, self.sc_ui[:2].type(torch.int64)), dim=-1)
        self.user_item_adj = self.get_norm_adj_mat_user_item_brif(ui_indices, uu_indices, self.mm_adj.cpu().coalesce().indices())
        self.user_item_adj = self.user_item_adj.to(self.device)

    def get_sc_mat_dataset_file(self, dataset, knn_k_uu, knn_k_ii, knn_k_ui, config):
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        out_path = f"{dataset_path}/LLM_generate_data/mat_file/"
        # user-user # item-item # user-item
        uu_path = out_path + f"ssc_omp_knn_C_top{knn_k_uu}_temp_user_user_indices_spa20_1e1_{dataset}.pt"
        ii_path = out_path + f"ssc_omp_knn_C_top{knn_k_ii}_temp_item_item_indices_spa20_1e1_{dataset}.pt"
        ui_path = out_path + f"ssc_omp_knn_C_top{knn_k_ui}_temp_user_item_indices_spa20_1e1_{dataset}.pt"

        uu_indices = torch.load(uu_path)
        ii_indices = torch.load(ii_path)
        ui_indices = torch.load(ui_path)

        return uu_indices, ii_indices, ui_indices

    def get_knn_adj_mat_ad_sc(self, mm_embeddings, sc_indices):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        indices = torch.cat((indices, sc_indices.to(self.device)[:2].type(torch.int64)), dim=-1)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def get_knn_adj_mat_user_item(self, user_embeddings, item_embeddings, knn):
        user_embeddings_norm = user_embeddings.div(torch.norm(user_embeddings, p=2, dim=-1, keepdim=True))  # M,d
        item_embeddings_norm = item_embeddings.div(torch.norm(item_embeddings, p=2, dim=-1, keepdim=True))  # N,d
        sim = torch.mm(user_embeddings_norm, item_embeddings_norm.transpose(1, 0))  # M,N
        _, knn_ind = torch.topk(sim, knn, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        # indices0 = torch.arange(knn_ind.shape[0])
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, knn)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def get_norm_adj_mat_user_item_brif(self, coo_matrix_ui, coo_matrix_user, coo_matrix_item):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = coo_matrix_ui
        data_dict = dict(zip(zip(inter_M[0].numpy(), inter_M[1].numpy() + self.n_users),
                             [1] * inter_M.shape[1]))
        data_dict.update(dict(zip(zip(inter_M[1].numpy() + self.n_users, inter_M[0].numpy()),
                                  [1] * inter_M.shape[1])))

        data_dict.update(dict(zip(zip(coo_matrix_user[0].numpy(), coo_matrix_user[1].numpy()),
                                  [1] * coo_matrix_user.shape[1])))
        data_dict.update(
            dict(zip(zip(coo_matrix_item[0].numpy() + self.n_users, coo_matrix_item[1].numpy() + self.n_users),
                     [1] * coo_matrix_item.shape[1])))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        feat_embedding = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        feat_list = [feat_embedding]
        for i in range(self.n_layers):
            tmp_embedding = torch.sparse.mm(self.user_item_adj, feat_embedding)
            feat_embedding = tmp_embedding
            feat_list += [feat_embedding]
        u_f_embeddings, i_f_embeddings = torch.split(feat_embedding, [self.n_users, self.n_items], dim=0)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings + u_f_embeddings, i_g_embeddings + i_f_embeddings, (
            u_g_embeddings, u_f_embeddings, i_g_embeddings, i_f_embeddings)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def cl_loss_alpha(self, embedings, uids, pos_items, neg_items, temp=0.2):
        main_user, feat_user, main_item, feat_item = embedings
        main_user = main_user / torch.sqrt(1e-8 + main_user.square().sum(-1, keepdim=True))
        feat_user = feat_user / torch.sqrt(1e-8 + feat_user.square().sum(-1, keepdim=True))
        main_item = main_item / torch.sqrt(1e-8 + main_item.square().sum(-1, keepdim=True))
        feat_item = feat_item / torch.sqrt(1e-8 + feat_item.square().sum(-1, keepdim=True))
        iids = torch.concat([pos_items, neg_items], dim=0)
        def get_cl(main, feat, id, temp):
            pos_ratings = torch.sum(main[id] * feat[id], dim=-1)  # b
            neg_ratings = torch.matmul(main[id], feat[id].T).squeeze(dim=1)  # b,N
            numerator = torch.exp(pos_ratings / temp)  # b
            denominator = torch.sum(torch.exp(neg_ratings / temp), dim=1)  # b
            ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))  # 1
            return ssm_loss
        user_cl = get_cl(main_user, feat_user, uids, temp)
        item_cl = get_cl(main_item, feat_item, iids, temp)
        loss_s = user_cl + item_cl
        return loss_s, user_cl, item_cl

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        ua_embeddings, ia_embeddings, cl_input = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                      neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        branch_cl_loss, user_cl, item_cl = self.cl_loss_alpha(cl_input, users, pos_items, neg_items, self.branch_cl_tau)

        # final loss
        emb_loss = (self.get_emb_loss_ad_neg_rand_infosty_out_mat(ia_embeddings, self.sc_item, num_neg=self.num_neg,
                                                             sem_tau=self.final_cl_tau)
                    + self.get_emb_loss_ad_neg_rand_infosty_out_mat(ua_embeddings, self.sc_user, num_neg=self.num_neg,
                                                               sem_tau=self.final_cl_tau)
                    + self.get_emb_loss_ui_neg_rand_infosty_out_mat(ua_embeddings, ia_embeddings, self.sc_ui,
                                                               num_neg=self.num_neg, sem_tau=self.final_cl_tau))

        return batch_mf_loss + self.reg_weight * (
                    mf_t_loss + mf_v_loss) + self.final_cl_reg * emb_loss + self.branch_cl_reg * branch_cl_loss

    def get_emb_loss_ad_neg_rand_infosty_out_mat(self, X, pos_indices, num_neg=1, sem_tau=0.2):
        X = F.normalize(X, dim=1)  # shape: (N, D)

        # 获取正样本向量
        x_i = X[pos_indices[0]]  # shape: (nnz, D)
        x_j = X[pos_indices[1]]  # shape: (nnz, D)

        # 正样本相似度
        positive_scores = torch.sum(x_i * x_j, dim=1)  # shape: (nnz,)

        # 负样本索引
        neg_col = torch.randint(0, X.shape[0], (pos_indices.shape[1] * num_neg,), device=X.device)
        neg_row = pos_indices[0].repeat_interleave(num_neg)

        x_i_neg = X[neg_row]  # shape: (nnz * num_neg, D)
        x_k_neg = X[neg_col]  # shape: (nnz * num_neg, D)
        neg_scores = torch.sum(x_i_neg * x_k_neg, dim=1)  # shape: (nnz * num_neg,)
        neg_scores = neg_scores.reshape((pos_indices.shape[1], num_neg))  # shape: (nnz, num_neg)

        # 损失计算
        numerator = torch.exp(positive_scores / sem_tau)
        denominator = numerator + torch.sum(torch.exp(neg_scores / sem_tau), dim=1)
        ssm_loss = torch.mean(-torch.log(numerator / denominator))
        return ssm_loss

    def get_emb_loss_ui_neg_rand_infosty_out_mat(self, X, Y, pos_indices, num_neg=1, sem_tau=0.2):
        X = F.normalize(X, dim=1)  # shape: (M, D)
        Y = F.normalize(Y, dim=1)  # shape: (N, D)

        # 获取正样本向量
        x_i = X[pos_indices[0]]  # shape: (nnz, D)
        x_j = Y[pos_indices[1]]  # shape: (nnz, D)

        # 正样本相似度
        positive_scores = torch.sum(x_i * x_j, dim=1)  # shape: (nnz,)

        # 负样本索引
        neg_col = torch.randint(0, Y.shape[0], (pos_indices.shape[1] * num_neg,), device=X.device)
        neg_row = pos_indices[0].repeat_interleave(num_neg)

        x_i_neg = X[neg_row]  # shape: (nnz * num_neg, D)
        x_k_neg = Y[neg_col]  # shape: (nnz * num_neg, D)
        neg_scores = torch.sum(x_i_neg * x_k_neg, dim=1)  # shape: (nnz * num_neg,)
        neg_scores = neg_scores.reshape((pos_indices.shape[1], num_neg))  # shape: (nnz, num_neg)

        # 损失计算
        numerator = torch.exp(positive_scores / sem_tau)
        denominator = numerator + torch.sum(torch.exp(neg_scores / sem_tau), dim=1)
        ssm_loss = torch.mean(-torch.log(numerator / denominator))
        return ssm_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e,_ = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

