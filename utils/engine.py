import os
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.loss_factory import Loss_factory
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.loss_factory import metric
from utils.utils import *
from utils.extra import *

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class Engine(object):
    def __init__(self, args, weights=None):
        self.args = args
        self.best_score = 0
        self.best_epoch = 0
        self.best_res = 0
        self.filename_best = None
        
        args.stage = 'test'
        self.model = get_model(args)
        if weights is not None:
            state_dict = torch.load(weights, weights_only=False)['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            print(f'Loading weights from {weights}')
        
        self.criterion = Loss_factory(args)
        self.optimizer = define_optimizer(args, self.model)
        self.scheduler = define_scheduler(args, self.optimizer)

    # def learning(self, model, dataset, criterion, optimizer, scheduler, fold, phase='train'):
    def learning(self, dataset):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        torch.cuda.empty_cache()

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train one epoch
            self.run_epoch(dataset, self.criterion, phase='train', optimizer=self.optimizer)
            # test one epoch
            result = self.run_epoch(dataset, self.criterion, phase='test')
            main_metric = 'AUC' if 'cls' in self.args.task else 'C-index'                
            score = result[main_metric]
            
            if score >= self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.best_res = result
                    
                
                full_state_dict = self.model.state_dict()
                filtered_state_dict = {
                    k: v for k, v in full_state_dict.items() 
                    if 'text_encoder' not in k and 'clin_encoder' not in k
                }
                
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': filtered_state_dict,
                    'best_score': score,
                    'score': result,
                    'fold': self.args.current_fold})
            print(' *** Current {}={:.4f}, best score={:.4f} at epoch {}'.format(main_metric, score, self.best_score, self.best_epoch))
            #self.scheduler.step()
            print('')
        return self.best_res

    def deploy(self, dataset):
        self.model = self.model.cuda()
        self.epoch = -1
        res = self.run_epoch(dataset, self.criterion, phase='test')
        return res
    
    def visualize(self, dataset):
        self.model = self.model.cuda()
        self.epoch = -1
        res = self.run_epoch(dataset, self.criterion, phase='test', visualize=True)
        return res
    
    def run_epoch(self, dataset, criterion, phase='train', optimizer=None, visualize=False):
        model_update = phase == 'train'
        if model_update:
            self.model.train()
        else:
            self.model.eval()
            
        dataset.phase = phase
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=model_update, num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_custom)
        num_samples = len(data_loader)
        
        sum_loss = 0.0
        all_loss_dict = {}
        for k in criterion.loss_collection.keys():
            all_loss_dict[k] = 0
            
        results = {}
        progressbar = tqdm(range(num_samples), desc='{} {} samples for epoch {}'.format(phase, num_samples, self.epoch), ncols=150)
        extra_save_dict = {}
        #for index, (data_dict, label_dict, pid) in zip(progressbar, data_loader):
        for index, (data_list, label_list, pid) in zip(progressbar, data_loader):
            
            for data_dict in data_list:
                for key, val in data_dict.items():
                    if isinstance(val, torch.Tensor):
                        data_dict[key] = val.cuda()
                        
            for label_dict in label_list:
                for key, val in label_dict.items():
                    if isinstance(val, torch.Tensor):
                        label_dict[key] = val.cuda()
            
            # 前向传播
            if phase == 'train':
                out_list = self.model(data_list, phase='train')
            else:
                with torch.no_grad():
                    out_list = self.model(data_list, phase='test')
            
            for input_data, label_data, out in zip(data_list, label_list, out_list):
                if self.args.task == 'cls':
                    label = label_data['label'].cuda()
                    out['probs'] = torch.softmax(out['pred'], dim=-1)
                    _, pred_cls = torch.max(out['probs'], dim=-1)
                    loss, loss_dict = criterion(out, {'label': label})
                    
                    results.setdefault('pred', []).append(pred_cls.detach().cpu().numpy()[0])
                    results.setdefault('probs', []).append(out['probs'].detach().cpu().numpy()[0])
                    results.setdefault('label', []).append(label.detach().cpu().numpy()[0])
                elif self.args.task == 'bcls':
                    label = label_data['label'].cuda().unsqueeze(0)
                    out['cls'] = torch.sigmoid(out['pred'])
                    pred_cls = (out['cls'] > 0.5).float()
                    loss, loss_dict = criterion(out, {'label': label})
                    
                    results.setdefault('pred', []).append(pred_cls.detach().cpu().numpy()[0])
                    results.setdefault('cls', []).append(out['cls'].detach().cpu().numpy()[0])
                    results.setdefault('label', []).append(label.detach().cpu().numpy()[0])
                else:
                    status = label_data['surv_status']
                    event_time = label_data['surv_time']
                    label = label_data['surv_label']
                    
                    out['hazards'] = torch.sigmoid(out['pred'])
                    out['S'] = torch.cumprod(1 - out['hazards'], dim=-1)
                    out['risk'] = -torch.sum(out['S'], dim=1)
                    loss, loss_dict = criterion(out, {'event_time': event_time.cuda(), 'c': status.cuda(), 'label': label.cuda()})
                                        
                    results.setdefault('risk', []).append(out['risk'].detach().cpu().numpy()[0])
                    results.setdefault('status', []).append(status.detach().cpu().numpy()[0])
                    results.setdefault('event_time', []).append(event_time.detach().cpu().numpy()[0])
                
            pats = []
            for k, v in loss_dict.items():
                all_loss_dict[k] += v
                pats.append('{}: {:.4f}'.format(k, all_loss_dict[k] / (index + 1)))
            str_loss = ', '.join(pats)
            sum_loss += loss.item()
            
            if model_update:
                loss.backward()
                #clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                lr_str = optimizer.param_groups[-1]['lr']
                progressbar.set_postfix_str('LR: {:.1e}, {}'.format(lr_str, str_loss))
                torch.cuda.empty_cache()
            elif visualize:
                extra_iter_save(self.args, self.model, data_dict, label_dict, out, pid, extra_save_dict)
                
        if visualize:
            extra_epoch_save(self.args, self.model, results, extra_save_dict)
            
        # calculate loss and error for epoch
        sum_loss /= len(progressbar)
        res = metric(results, self.args.task)
        print('loss: {:.4f}, {}'.format(sum_loss, res))
        return res

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.args.results_dir, f"fold_{str(state['fold'])}_epoch{state['epoch']}_{state['best_score']:.4f}.pth.tar")
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)

def collate_custom(batch):
    '''
    collate函数保持不变
    '''
    feats = [item[0] for item in batch] 
    labels = [item[1] for item in batch] 
    pids = [item[2] for item in batch]
    return feats, labels, pids