import os
import torch
import wandb
import numpy as np

from args import parse_args
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from dkt.dataloader import Preprocess
from dkt import trainer
from dkt.utils import setSeeds


def main(args):
    # wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args.model_dir = os.path.join(args.model_dir, f'{args.model}_{args.info}')

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    get_train_oof(args, train_data)


def get_train_oof(args, data, fold_n=10, stratify=True):
        if stratify:
            kfold = StratifiedKFold(n_splits=fold_n)
        else:
            kfold = KFold(n_splits=fold_n)

        # 클래스 비율 고려하여 Fold별로 데이터 나눔
        target = get_target(data)

        for i, (train_index, valid_index) in enumerate(kfold.split(data, target)):
            print(f'Calculating train oof {i + 1}')

            train_data, valid_data = data[train_index], data[valid_index]

            wandb.init(project=f'dkt_kfold_{args.model}', config=vars(args))
            wandb.run.name = f'{args.model}_{args.info}_{i+1}'
            wandb.run.save()
            args.kfold = i
           
            trainer.run(args, train_data, valid_data)
            

def get_target(datas):
        targets = []
        for data in datas:
            targets.append(data[0][-1])

        return np.array(targets)

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)