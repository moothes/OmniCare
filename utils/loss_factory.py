import torch
import math 
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score, f1_score, auc, classification_report, precision_recall_fscore_support, confusion_matrix

def nll_loss(hazards, S, Y, c, alpha=0., eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    #print(S_padded)
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)   
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def nllsurv(out, gt, alpha=0.):
    #loss = 0
    #for haz, s in zip(out['hazards'], out['S']):
    #    loss += nll_loss(haz, s, gt['label'], gt['c'], alpha=alpha)
    loss = nll_loss(out['hazards'], out['S'], gt['label'], gt['c'], alpha=alpha)
    return loss
    

def CLSLoss1(out, gt, alpha=0.):
    loss = 0
    for pred in out['pred1']:
        loss += F.cross_entropy(pred, gt['label'].long())
    #print(out['pred'], lbl)
    return loss

def CLSLoss(out, gt, alpha=0.):
    #loss = 0
    #for pred in out['pred']:
    #print(out['pred'].shape, gt['label'].shape)
    #lbl = F.one_hot(gt['label'].long(), num_classes=5)
    loss = F.cross_entropy(out['cls'], gt['label'].long())
    #print(out['pred'], lbl)
    return loss
    
def BCLSLoss(out, gt, alpha=0.):
    #loss = 0
    #for pred in out['pred']:
    #print(out['cls'].shape, gt['label'].shape)
    #lbl = F.one_hot(gt['label'].long(), num_classes=5)
    loss = F.binary_cross_entropy(out['cls'], gt['label'].float()).mean()
    #print(out['pred'], lbl)
    return loss

def SIMLoss(out, gt, alpha=0.):
    lf, rf = out['align']
    loss = 1 - F.cosine_similarity(lf, rf).mean()
    #print(lf.shape, rf.shape, loss.shape)
    return loss

def ReembLoss(out, gt, alpha=0.):
    return out['reemb']
    
def contrastive(feats, labels, temperature=0.1):
    #feats, labels = out['contrastive']    # feats shape: [B, D]
    feats = torch.concat(feats, dim=0)
    labels = torch.tensor(labels).cuda()
    #labels = outputs['labels']    # labels shape: [B]
    #print(feats.shape, labels.shape)

    feats = F.normalize(feats, dim=-1, p=2)

    logits_mask = torch.eye(feats.shape[0]).float().cuda()
    mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).float() - logits_mask

    # compute logits
    logits = torch.matmul(feats, feats.T) / temperature
    logits = logits - logits_mask * 1e9

    # optional: minus the largest logit to stablize logits
    #logits = stablize_logits(logits)
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()

    # compute ground-truth distribution
    p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
    #loss = compute_cross_entropy(p, logits)
    logits = F.log_softmax(logits, dim=-1)
    loss = torch.sum(p * logits, dim=-1)

    return -loss.mean()

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def topk_cluster(feature,supports,scores,p,k=3):
    #p: outputs of model batch x num_class
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    topk_sim_matrix,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near].detach().clone()  #batch x K x num_class
    #print(p.shape, scores_near.shape)
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near)**2,-1)
    
    loss = -1.0* topk_sim_matrix * diff_scores
    return loss.mean()
    
def select_supports(ent_s, y_hat, filter_K=100, n_cls=5):
    #ent_s = self.ent
    y_hat = y_hat.argmax(dim=1).long()
    #filter_K = self.filter_K
    if filter_K == -1:
        indices = torch.LongTensor(list(range(len(ent_s))))

    indices = []
    indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
    for i in range(n_cls):
        #print(y_hat.shape, ent_s.shape)
        _, indices2 = torch.sort(ent_s[y_hat == i])
        indices.append(indices1[y_hat==i][indices2][:filter_K])
    indices = torch.cat(indices)

    return indices
    #self.supports = self.supports[indices]
    #self.labels = self.labels[indices]
    #self.ent = self.ent[indices]
    #self.scores = self.scores[indices]
    
    #return self.supports, self.labels

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div        
        

def prototype_loss(z,p,labels=None,use_hard=False,tau=1):
    #z [batch_size,feature_dim]
    #p [num_class,feature_dim]
    #labels [batch_size,]        
    z = F.normalize(z,1)
    p = F.normalize(p,1)
    dist = z @ p.T / tau
    if labels is None:
        _,labels = dist.max(1)
    if use_hard:
        """use hard label for supervision """
        #_,labels = dist.max(1)  #for prototype-based pseudo-label
        labels = labels.argmax(1)  #for logits-based pseudo-label
        loss =  F.cross_entropy(dist,labels)
    else:
        """use soft label for supervision """
        #print(labels.shape, dist.shape)
        loss = softmax_kl_loss(labels.detach(),dist).sum(1).mean(0)  #detach is **necessary**
        #loss = softmax_kl_loss(dist,labels.detach()).sum(1).mean(0) achieves comparable results
    return dist,loss

def attloss(out, gt, alpha=0.):
    #print(out['att'].shape)
    loss = (out['att'].softmax(1) * out['att'].log_softmax(1)).sum(1).mean()
    return loss
    
def tentloss(out, gt, alpha=0.):
    #print(out['pred'].shape)
    loss = softmax_entropy(out['pred']).mean()
    return loss
    
    
def pseloss(out, gt, alpha=0.):
    
    loss = F.binary_cross_entropy(out['cls'], out['pse']).mean()
    #loss = softmax_entropy(out['pred']).mean()
    return loss
    
def tsdloss(out, gt, n_cls=5):
    loss = 0
    z = out['feature']
    p = out['pred']

    yhat = F.one_hot(p.argmax(1), num_classes=n_cls).float()
    yent = softmax_entropy(p)
    yscore = F.softmax(p,1)

    with torch.no_grad():
        supports = torch.zeros((0, 256)).cuda()
        labels = torch.zeros((0, n_cls)).cuda()
        ents = torch.zeros((0)).cuda()
        scores = torch.zeros((0)).cuda()
        for label in range(5):
            feats = torch.cat(feat_bank[label], dim=0)
            logits = torch.cat(logit_bank[label], dim=0)
            
            label = F.one_hot(logits.argmax(1), num_classes=n_cls).float()
            ent = softmax_entropy(logits)
            score = F.softmax(logits,1)
            #print(prob)
            #att = 1.5 + torch.sum(prob * torch.log(prob + 1e-10), dim=-1, keepdims=True)
            
            supports = torch.cat([supports, feats], dim=0)
            labels = torch.cat([labels, label], dim=0)
            ents = torch.cat([ents, ent], dim=0)
            scores = torch.cat([scores, logits], dim=0)
        
        supports = torch.cat([supports, z], dim=0)
        labels = torch.cat([labels, yhat], dim=0)
        ents = torch.cat([ents, yent], dim=0)
        scores = torch.cat([scores, yscore], dim=0)
        
        indices = select_supports(ents, labels, filter_K=100, n_cls=n_cls)
        
        supports = supports[indices]
        labels = labels[indices]
        ents = ents[indices]
        scores = scores[indices]
        
        supports = F.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
    dist,loss = prototype_loss(z,weights.T,yscore,use_hard=False)
    loss += topk_cluster(z.detach().clone(),supports,scores,p,k=3)
    #loss += F.cross_entropy(out['pred'], yhat).mean() * 2
    return loss
    

def ttaloss(out, gt, n_cls=5):
    loss = 0
    z = out['feature']
    p = out['pred']

    yhat = F.one_hot(p.argmax(1), num_classes=n_cls).float()
    yent = softmax_entropy(p)

    with torch.no_grad():
        supports = torch.zeros((0, 256)).cuda()
        labels = torch.zeros((0, n_cls)).cuda()
        ents = torch.zeros((0)).cuda()
        for label in range(5):
            feats = torch.cat(feat_bank[label], dim=0)
            logits = torch.cat(logit_bank[label], dim=0)
            
            label = F.one_hot(logits.argmax(1), num_classes=n_cls).float()
            ent = softmax_entropy(logits)
            #print(prob)
            #att = 1.5 + torch.sum(prob * torch.log(prob + 1e-10), dim=-1, keepdims=True)
            
            supports = torch.cat([supports, feats], dim=0)
            labels = torch.cat([labels, label], dim=0)
            ents = torch.cat([ents, ent], dim=0)
        
        supports = torch.cat([supports, z], dim=0)
        labels = torch.cat([labels, yhat], dim=0)
        ents = torch.cat([ents, yent], dim=0)
        
        indices = select_supports(ents, labels, filter_K=100, n_cls=n_cls)
        
        supports = supports[indices]
        labels = labels[indices]
        ents = ents[indices]
        
        supports = F.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
    loss += (z @ F.normalize(weights, dim=0)).mean()
    return loss

class Loss_factory(nn.Module):
    def __init__(self, args):
        super(Loss_factory, self).__init__()
        loss_item = args.loss.split(',')
        self.loss_collection = {}
        for loss_im in loss_item:
            tags = loss_im.split('_')
            self.loss_collection[tags[0]] = float(tags[1]) if len(tags) == 2 else 1.
            
        
    def forward(self, preds, target):
        loss_sum = 0
        ldict = {}
        for loss_name, weight in self.loss_collection.items():
            loss = eval(loss_name + '(preds, target) * weight')
            ldict[loss_name] = loss
            loss_sum += loss
        return loss_sum, ldict

def metric(results, task):
    if task == 'cls':
        all_cls = []
        all_lbls = []
        all_scores = []
        for res in results:
            print(res.shape)
            all_scores.append(res[:-2])
            all_cls.append(res[-2:-1])
            all_lbls.append(res[-1:])
        
        return cls_metric(np.concatenate(all_cls, axis=0), np.concatenate(all_lbls, axis=0), np.concatenate(all_scores, axis=0))
    elif task == 'bcls':
        #all_cls = []
        #all_lbls = []
        #all_scores = []
        #for res in results:
        #    all_scores.append(res[0])
        #    all_cls.append(res[1])
        #    all_lbls.append(res[2])
        all_cls = np.array(results['cls'])
        all_pred = np.array(results['pred'])
        all_label = np.array(results['label'])

        
        return cls_metric(all_pred, all_label, all_cls)
    else:
        out_dict = {}

        all_risk_concatenated = np.array(results['risk'])
        all_status_concatenated = np.array(results['status'])
        all_time_concatenated = np.array(results['event_time'])

        try:
            #print('censored samples: ', np.sum(all_status_concatenated))
            cindex = concordance_index_censored(
                (1 - all_status_concatenated).astype(bool),
                all_time_concatenated,
                all_risk_concatenated,
                tied_tol=1e-08
            )[0]
        except Exception as e:
            print(f"Error calculating C-index: {e}")
            cindex = 0.0
        out_dict['C-index'] = cindex
        
        # Bootstrap
        n_bootstraps = 1000
        cindex_scores = []

        for _ in range(n_bootstraps):
            #indices = np.random.choice(len(all_time_concatenated), len(all_time_concatenated), replace=True)
            
            aa = 0
            fail = False
            while np.sum(aa) == 0 or fail:
                indices = np.random.choice(len(all_time_concatenated), len(all_time_concatenated), replace=True)
                aa = (1 - all_status_concatenated).astype(bool)[indices]
            
                bb = all_time_concatenated[indices]
                cc = all_risk_concatenated[indices]
            
                try:
                    cindex_bootstrap = concordance_index_censored(aa, bb, cc, tied_tol=1e-08)[0]
                    cindex_scores.append(cindex_bootstrap)
                    fail = False
                except Exception as e:
                    #print(f"Warning: Bootstrap iteration failed: {e}")
                    fail = True
                    #continue

        if cindex_scores:
            out_dict['95%CI'] = np.percentile(cindex_scores, [2.5, 97.5])
            
        return out_dict
        

def cls_metric(pred, label, score):
    acc = np.sum(pred == label) / len(label)
    auc_score = roc_auc_score(label, score, average=None, multi_class='ovr')
    f1score = f1_score(label, pred, average='weighted')
    
    n_bootstraps = 1000

    # Bootstrap to calculate the confidence interval
    auc_scores = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(len(label)), len(label), replace=True)
        while label[indices].sum() == 0 or label[indices].sum() == len(label):
            indices = np.random.choice(range(len(label)), len(label), replace=True)

        auc_bootstrap = roc_auc_score(label[indices], score[indices], multi_class='ovr')
        auc_scores.append(auc_bootstrap)

    # Calculate the 95% confidence interval
    confidence_interval = np.percentile(auc_scores, [2.5, 97.5])

    res = {'ACC': round(acc, 4),
           'AUC': round(np.mean(auc_score), 4),
           '95CI': confidence_interval,
           'F1score': round(f1score, 4),
          }
    return res