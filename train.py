import os
import torch
import wandb

from args import parse_args
from sklearn.linear_model import LinearRegression

from dkt.dataloader import Preprocess
from dkt import trainer
from dkt.utils import setSeeds


def main(args):
    # wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data, seed=42)

    wandb.init(project=f'dkt_{args.model}', config=vars(args), name=f'{args.model}_{args.info}')
    args.kfold = 9

    trainer.run(args, train_data, valid_data)
    
if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)