import os
import torch
import numpy as np
from tqdm import tqdm
import copy
import random

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.loss_factory import Loss_factory, nllsurv, contrastive
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.loss_factory import metric
from utils.utils import *
from utils.extra import *
from torch.nn import functional as F

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class PretrainEngine(object):
    def __init__(self, args):
        self.args = args
        self.best_score = float('inf')
        self.best_epoch = 0
        self.best_res = 0
        self.filename_best = None
        
        self.model = get_model(args)
        
        self.criterion = Loss_factory(args)
        self.optimizer = define_optimizer(args, self.model)
        self.scheduler = define_scheduler(args, self.optimizer)
        self.global_iteration = 0
        
        if args.resume != '':
            state_dict = torch.load(args.resume, weights_only=False)
            self.model.load_state_dict(state_dict['state_dict'], strict=False)
            
            
            optimizer_state = state_dict['optimizer']
            #current_state = self.optimizer.state_dict()
            #filtered_state = {k: v for k, v in optimizer_state.items() if k in current_state}

            # 加载过滤后的state
            self.optimizer.load_state_dict(optimizer_state)  # 现在使用strict=True
            #self.optimizer.load_state_dict(state_dict['optimizer'], strict=False)
            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            print(f'Loading weights from {args.resume}')
            
            self.global_iteration = int(args.resume.split('/')[-1].split('_')[1][4:])
        # 新增：迭代计数器

    def learning(self, dataset):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        torch.cuda.empty_cache()

        # 获取数据加载器
        data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=16, pin_memory=True, drop_last=False, collate_fn=collate_custom)
        
        total_iterations = self.args.num_epoch * len(data_loader)
        
        # 初始化优化器梯度
        self.optimizer.zero_grad()
        
        # 设置梯度累积步数
        accumulation_steps = 8
        
        # 按iteration训练
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            dataset.phase = 'train'
            self.model.train()
            
            num_samples = len(data_loader)
            progressbar = tqdm(range(num_samples), desc=f'Epoch {epoch} training', ncols=150)
            
            sum_loss = 0.0
            all_loss_dict = {}
            extra_save_dict = {}
            
            for index, (data_list, label_list, pid) in zip(progressbar, data_loader):
                # 数据预处理（与原来相同）
                for data_dict in data_list:
                    for key, val in data_dict.items():
                        if isinstance(val, torch.Tensor):
                            data_dict[key] = val.cuda()
                            
                for label_dict in label_list:
                    for key, val in label_dict.items():
                        if isinstance(val, str) or isinstance(val, list):
                            continue
                        label_dict[key] = val.cuda()
                
                # 前向传播
                out_list = self.model(data_list, phase='train')
                
                # 计算loss（与原来相同的逻辑）
                sup_loss = torch.tensor(0.0, device='cuda')
                for input_data, label_data, out_data in zip(data_list, label_list, out_list):
                    # ... 原有的loss计算代码保持不变 ...
                    # （这里省略完整的loss计算，保持和原来一样）
                    if 'surv_label' in label_data.keys():
                        new_out = {}
                        status = label_data['surv_status'].cuda().view(1)
                        event_time = label_data['surv_label'].cuda().view(1)
                        new_out['hazards'] = torch.sigmoid(out_data['os'])
                        new_out['S'] = torch.cumprod(1 - new_out['hazards'], dim=-1)
                        new_out['risk'] = -torch.sum(new_out['S'], dim=1)
                        os_loss = nllsurv(new_out, {'label': event_time, 'c': status})
                        sup_loss += os_loss
                        all_loss_dict.setdefault('os', []).append(os_loss.item())
                        
                    if 'ihc_pt' in label_data.keys():
                        IHC_label = label_data['ihc_pt']
                        mask = IHC_label.gt(-1)
                        IHC_label = torch.clip(IHC_label, 0, 1)
                        ihc_loss_ = F.binary_cross_entropy_with_logits(out_data['diag'], IHC_label.unsqueeze(0).float(), reduction='none') * mask.float()
                        ihc_loss = ihc_loss_.sum() / (mask.sum() + 1e-10)
                        sup_loss += ihc_loss
                        all_loss_dict.setdefault('diag', []).append(ihc_loss.item())
                    
                    reco_loss = 0
                    if 'text_pt' in label_data.keys():
                        text_label = label_data['text_pt']
                        text_pred = out_data['text']
                        reco_loss += F.mse_loss(text_pred, text_label)
                        #reco_loss += 1 - (F.cosine_similarity(text_label, text_pred, dim=-1)).mean()
                    if 'clin_pt' in label_data.keys():
                        clin_label = label_data['clin_pt']
                        clin_pred = out_data['clin']
                        reco_loss += F.mse_loss(clin_pred, clin_label)
                        #reco_loss += 1 - (F.cosine_similarity(clin_label, clin_pred, dim=-1)).mean()
                    
                    if 'gene' in input_data.keys():
                        gene_label = input_data['gene']
                        gene_pred = out_data['gene']
                        reco_loss += F.mse_loss(gene_pred, gene_label)
                        #reco_loss += 1 - (F.cosine_similarity(gene_label, gene_pred, dim=-1)).mean()
                    
                    if out_data['reco_he'] is not None:
                        he_label = out_data['reco_he']
                        he_pred = out_data['he']
                        reco_loss += F.mse_loss(he_pred, he_label)
                        #reco_loss += 1 - (F.cosine_similarity(he_label, he_pred, dim=-1)).mean()
                    if out_data['reco_ihc'] is not None:
                        ihc_label = out_data['reco_ihc']
                        ihc_pred = out_data['ihc']
                        reco_loss += F.mse_loss(ihc_pred, ihc_label)
                        #reco_loss += 1 - (F.cosine_similarity(ihc_label, ihc_pred, dim=-1)).mean()
                            
                    all_loss_dict.setdefault('reco', []).append(reco_loss.item())
                    sup_loss += reco_loss.item()
                
                # 无监督对比学习
                unsup_loss = torch.tensor(0.0, device='cuda')
                feats = []
                tag = []
                for iddx, out_data in enumerate(out_list):
                    for feat in out_data['modality_feats']:
                        feats.append(feat)
                        tag.append(iddx)
                unsup_loss += contrastive(feats, tag)
                all_loss_dict.setdefault('conts', []).append(unsup_loss.item())

                loss = sup_loss + unsup_loss
                
                # 梯度累积
                loss = loss / accumulation_steps
                loss.backward()
                
                # 更新迭代计数器
                self.global_iteration += 1
                
                # 每accumulation_steps步更新一次参数
                if self.global_iteration % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 计算平均loss用于显示
                str_loss = ', '.join([f'{k}: {np.mean(v):.4f}' for k, v in all_loss_dict.items()])
                sum_loss += loss.item() * accumulation_steps  # 恢复真实的loss值
                
                lr_str = self.optimizer.param_groups[-1]['lr']
                progressbar.set_postfix_str(f'Iter: {self.global_iteration}, LR: {lr_str:.1e}, {str_loss}')
                
                # ========== 每1000次迭代保存weights ==========
                if self.global_iteration % 5000 == 0:
                    self.save_checkpoint_by_iteration(self.global_iteration, loss.item() * accumulation_steps)
                
                torch.cuda.empty_cache()
                extra_iter_save(self.args, self.model, data_dict, label_dict, out_list, pid, extra_save_dict)
            
            # 处理最后剩余的梯度
            if self.global_iteration % accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # epoch结束时的统计
            avg_loss = sum_loss / num_samples
            print(f' *** Epoch {epoch} completed, Average loss={avg_loss:.4f} at iteration {self.global_iteration}')
            print('')
            
            # 每个epoch结束后保存一次（可选）
            self.save_checkpoint({
                'epoch': epoch,
                'iteration': self.global_iteration,
                'state_dict': self.model.state_dict(),
                'best_score': avg_loss
            })
            
        return avg_loss

    def deploy(self, data_loader, criterion):
        res = self.run_epoch(data_loader, criterion, phase='test')
        return res
    
    def save_checkpoint_by_iteration(self, iteration, loss):
        """按iteration保存模型"""
        filename = os.path.join(self.args.results_dir, f"model_iter{iteration}_loss{loss:.4f}.pth.tar")
        print(f'Saving checkpoint at iteration {iteration} with loss {loss:.4f}')
        full_state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items() 
            if 'text_encoder' not in k and 'clin_encoder' not in k
        }
        state = {
            'iteration': iteration,
            'state_dict': filtered_state_dict,
            'loss': loss,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, filename)
        return filename

    def save_checkpoint(self, state):
        """原有的保存函数，用于保存epoch级别的checkpoint"""
        # 过滤掉text_encoder和clin_encoder
        full_state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v for k, v in full_state_dict.items() 
            if 'text_encoder' not in k and 'clin_encoder' not in k
        }
        state['state_dict'] = filtered_state_dict
        
        filename = os.path.join(self.args.results_dir, f"model_epoch{state['epoch']}_iter{state['iteration']}_loss{state['best_score']:.4f}.pth.tar")
        print(f'Save checkpoint: {filename}')
        torch.save(state, filename)
        return filename

def collate_custom(batch):
    '''
    collate函数保持不变
    '''
    feats = [item[0] for item in batch] 
    labels = [item[1] for item in batch] 
    pids = [item[2] for item in batch]
    return feats, labels, pids