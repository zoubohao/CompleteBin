
import numpy as np
import torch
import torch.nn.functional as F


def info_nce_loss_2_views(features, batch_size, temperature, device):
    """_summary_

    Args:
        features (_type_): [view_11, ..., view_1b, view_21, ..., view_2b, ..., view_n1, ..., view_nb]
        batch_size (_type_): _description_
        n_views (_type_): _description_
        temperature (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_views = 2
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    ### split matrix and compare with each other
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    return logits, labels


def info_nce_loss(features, batch_size, n_views, temperature, device, criterion):
    """_summary_

    Args:
        features (_type_): [view_11, ..., view_1b, view_21, ..., view_2b, ..., view_n1, ..., view_nb]
        batch_size (_type_): _description_
        n_views (_type_): _description_
        temperature (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=-1)
    similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(-1, 1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    negatives = negatives[:, None].expand(-1, n_views - 1, -1).flatten(0, 1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    return criterion(logits, labels), logits, labels


def disp_info_nce_loss(features, batch_size, n_views, temperature, device):
    """_summary_

    Args:
        features (_type_): [view_11, ..., view_1b, view_21, ..., view_2b, ..., view_n1, ..., view_nb]
        batch_size (_type_): _description_
        n_views (_type_): _description_
        temperature (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=-1)
    similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    negatives = negatives[:, None].expand(-1, n_views - 1, -1).flatten(0, 1) / temperature
    return torch.log(torch.mean(torch.exp(negatives)))


def info_nce_loss_for_loop(features, batch_size, n_views, temperature, device, criterion):
    n_views_emb = torch.chunk(features, chunks=n_views, dim=0)
    logits_list = []
    labels_list = []
    loss_simclr = 0.
    for i in range(n_views):
        subloss = 0
        cur_view = n_views_emb[i]
        for v in range(i + 1, n_views):
            other_view = n_views_emb[v]
            cat_two_view = torch.cat([cur_view, other_view], dim=0)
            logits, labels = info_nce_loss_2_views(cat_two_view, batch_size, temperature, device)
            logits_list.append(logits)
            labels_list.append(labels)
            subloss += criterion(logits, labels)
        subloss /= n_views - i
        loss_simclr += subloss
    loss_simclr /= n_views
    logits_cat = torch.cat(logits_list, dim=0)
    labels_cat = torch.cat(labels_list, dim=0)
    return loss_simclr, logits_cat, labels_cat


### Hard Negative Sample Loss for SimClR
def info_nce_loss_HNS_2_views(out_1, out_2, batch_size, temperature, device, tau_plus, beta = 1.0):
    
    def get_negative_mask(batch_size):
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        negative_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        return negative_mask   
    
    # neg score
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)
    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    
    N = batch_size * 2 - 2
    imp = (beta * neg.log()).exp()
    reweight_neg = (imp * neg).sum(dim = -1) / imp.mean(dim = -1)
    Ng = (- tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
    Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss


def info_nce_loss_HNS_for_loop(features, batch_size, n_views, temperature, device, tau_plus, beta):
    n_views_emb = torch.chunk(features, chunks=n_views, dim=0)
    loss_simclr = 0.
    for i in range(n_views):
        cur_view = n_views_emb[i]
        for v in range(i + 1, n_views):
            other_view = n_views_emb[v]
            subloss = info_nce_loss_HNS_2_views(cur_view, other_view, batch_size, temperature, device, tau_plus, beta)
        loss_simclr += subloss
    loss_simclr /= n_views
    return loss_simclr


def info_nce_loss_mix_up(
    ori_features, 
    mix_features,
    ori_lam,
    other_lam,
    batch_size, 
    temperature,
    device, 
    criterion):
    other_seq_emb, mix_view_emb = torch.chunk(mix_features, chunks=2, dim=0)
    labels = torch.arange(batch_size).to(device)
    # mix 2 ori
    mix2ori_logits = torch.matmul(mix_view_emb, ori_features.T) / temperature # [b, b]
    mix2ori_loss = criterion(mix2ori_logits, labels) * ori_lam
    # mix 2 other
    mix2oth_logits = torch.matmul(mix_view_emb, other_seq_emb.T) / temperature # [b, b]
    mix2oth_loss = criterion(mix2oth_logits, labels) * other_lam
    # print(f"mix2ori_loss shape: {mix2ori_loss.shape}, mix2oth_loss shape: {mix2oth_loss.shape}, ori_lam {ori_lam.shape}, {ori_lam}, {other_lam}")
    loss_mix = torch.mean(mix2ori_loss + mix2oth_loss)
    logits_cat = torch.cat([mix2ori_logits, mix2oth_logits], dim=0)
    labels_cat = torch.cat([labels, labels.clone()], dim=0)
    return loss_mix, logits_cat, labels_cat


### useless
@torch.no_grad()
def distributed_sinkhorn(
        out,
        epsilon,
        sinkhorn_iterations):
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1]  # number of samples to assign
    K = Q.shape[0]  # how many prototypes
    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def swav_loss(
    prototypes_output,
    n_views,
    batch_size,
    temperature
):
    loss_swav = 0
    for crop_id in range(n_views):
        with torch.no_grad():
            out = prototypes_output[batch_size * crop_id: batch_size * (crop_id + 1)].detach()
            # get assignments
            q = distributed_sinkhorn(out, 0.05, 3)[-batch_size:]
        # cluster assignment prediction
        subloss = 0
        for v in np.delete(np.arange(n_views), crop_id):
            x = prototypes_output[batch_size * v: batch_size * (v + 1)] / temperature
            subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
        loss_swav += subloss / (n_views - 1)
    loss_swav /= n_views
    return loss_swav


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vecreg_loss(
    features_x_view,
    features_y_view,
    batch_size,
    sim_coeff=25.,
    std_coeff=25.,
    cov_coeff=1.,
):
    num_features = features_x_view.size(-1)
    x = features_x_view
    y = features_y_view
    repr_loss = F.mse_loss(x, y)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)
    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss
