import os
import importlib
import time



def get_save_path(args):
    #results_dir = "./results/{model}/{}_{task}_{per}_{time}".format(model=args.model, task=args.task_config, per=args.per, time=time.strftime("%Y-%m-%d]-[%H-%M-%S"))
    results_dir = f"./results/{args.model}/{args.stage}/{time.strftime('%m%d-%H%M%S')}_{args.per}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    return results_dir
    #args.results_dir = results_dir
    #csv_path = os.path.join(results_dir, "results_all.csv")

def get_model(args):
    return importlib.import_module('models.{}.network'.format(args.model)).Network(args)