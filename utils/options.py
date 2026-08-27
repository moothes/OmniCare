import argparse
import os

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for multimodal AI model for precision oncology")
    parser.add_argument("model", type=str, default="OmniCare", help="Model name")
    
    # Model Parameters.
    parser.add_argument("--modal", type=lambda x: x.split(','), default=["text", "gene", "path", "clin", "ihc"], help="Input modalities")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible experiment (default: 0)")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--num_subsets", type=int, default=30, help="Number of cancer types")
    parser.add_argument("--gene_length", type=int, default=768, help="Length of gene vector")
    

    # Experiment Parameters
    parser.add_argument("--folds", type=lambda x: x.split(','), default="0,1,2,3,4", help="Number of fold")
    parser.add_argument("--stage", type=str, choices=["pretrain", "finetune", "test", "external"], default="pretrain")
    parser.add_argument("--gpu", type=str, default="0", help="Which GPU used for training")
    parser.add_argument("--num_epoch", type=int, default=20, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test"], help="Phase during running models")
    
    # Training Parameters
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead", "SGD"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--loss", type=str, default="nllsurv", help="slide-level classification loss function (default: ce)")
    parser.add_argument("--resume", type=str, default="", help="Which GPU used for training")
    
    # Test Parameters
    parser.add_argument("--cohort", type=str, default="", help="Define cohort and endpoint")
        
    # Training Percentage in Data Efficiency Experiment
    parser.add_argument("--per", type=float, default=1, help="Maximum number of epochs to train (default: 20)")
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #args.model = model_name
    #args.task_config = task_name
    
    if args.cohort:
        args.dataset_name, args.endpoint = args.cohort.split('_')
        if args.endpoint in ['os', 'dfs', 'dss', 'dm', 'lr', 'bcr', 'pfs']:
            args.task = 'surv'
            args.loss = 'nllsurv'
            args.n_classes = 5
        elif args.endpoint in ['os5y', 'dss5y', 'pcr', 'pcrp']:
            args.task = 'bcls'
            args.loss = 'BCLSLoss'
            args.n_classes = 1
            #self.subtype_list = range(self.args.n_classes) #self.args.subtype_list.split(',')
        elif args.endpoint in ['']:
            args.task = 'cls'
            args.loss = 'CLSLoss'
        else:
            raise ValueError(f"Unknown endpoint: {args.endpoint}")
    
    return args
