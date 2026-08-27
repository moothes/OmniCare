import os
import torch
import numpy as np
import pandas as pd

def extra_iter_save(args, model, data, label, pred, pid, extra_save_dict={}):
    #print(pid, pred['router'])
    
    for pidd, router in zip(pid, pred):
        save_path = f'path_att/{pidd.split("/")[1]}.npy'
        if 'path_att' in router.keys():
            router = np.array(router['path_att'].detach().cpu().numpy())
            np.save(save_path, router)
        #print(pidd.split('/')[1], router['router'])
    '''
    for pidd, router in zip(pid, pred):
        save_path = f'router/{pidd.split("/")[1]}.npy'
        router = np.array(router['router'].detach().cpu().numpy())
        np.save(save_path, router)
        #print(pidd.split('/')[1], router['router'])
    '''
    return 

    res_csv_file = os.path.join('visual', args.cohort, 'prediction.csv')
    if not os.path.exists(res_csv_file):
        df = pd.DataFrame(columns=['pid', 'risk', 'status', 'event_time'])
    else:
        df = pd.read_csv(res_csv_file)
    
    new_row = pd.DataFrame({'pid': pid, 'risk': pred['risk'].detach().cpu(), 'status': label['surv_status'].detach().cpu(), 'event_time': label['surv_time'].detach().cpu()})
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv(res_csv_file, index=False)

def extra_epoch_save(args, model, results, extra_save_dict={}):
    return

    if 'cls' in args.task:
        print(f"Extra epoch save: {results}.")
        
        save_dir = os.path.join('visual', args.cohort, f'{args.model}_res')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        pred_vector = np.concatenate(results['pred']).flatten()
        cls_vector = np.concatenate(results['cls']).flatten()
        label_vector = np.concatenate(results['label']).flatten()
        
        np.savez(os.path.join(save_dir, f"{args.model}_{args.cohort}_{args.current_fold}_pred.npz"), pred=pred_vector, cls=cls_vector, label=label_vector)
        
        #print(pred_vector.shape, cls_vector.shape, label_vector.shape)
    else:
        print(results.keys())
        save_dir = os.path.join('visual', f'{args.cohort}_mis', f'{args.model}_res')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        risk_vector = np.array(results['risk']).flatten()
        status_vector = np.array(results['status']).flatten()
        time_vector = np.array(results['event_time']).flatten()
        
        print(risk_vector.shape, status_vector.shape, time_vector.shape)
        
        np.savez(os.path.join(save_dir, f"{args.model}_{args.cohort}_{args.current_fold}_pred.npz"), risk=risk_vector, status=status_vector, event_time=time_vector)