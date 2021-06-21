import os
import copy

import wandb
import torch
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, Saint, TfixupBert


# feature 추가 #1~4 수정
def run(args, train_data, valid_data):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args)

    wandb.watch(model)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        wandb.log({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_auc": train_auc, 
            "train_acc":train_acc,
            "valid_auc":auc, 
            "valid_acc":acc,
            })

        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, f'{args.model}_{args.info}_{args.kfold}.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        inputs = process_batch(batch, args)
        preds = model(inputs)
        targets = inputs[-1] # correct


        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc} ') #LR : {optimizer.get}
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        inputs = process_batch(batch, args)

        preds = model(inputs)
        targets = inputs[-1] # correct


        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets



def inference(args, test_data):
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        inputs = process_batch(batch, args)

        preds = model(inputs)
        
        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, f"{args.model_name[:-3]}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
        
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))


def multi_inference(args, test_data):
    model_list = os.listdir(args.model_dir)
    print(model_list)

    kfold_preds = np.zeros(744)
    for model in model_list:
        args.model_name = model
        model = load_model(args)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)
        
        total_preds = []
        for step, batch in enumerate(test_loader):
            inputs = process_batch(batch, args)

            preds = model(inputs)
            
            # predictions
            preds = preds[:,-1]
            
            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()
                
            total_preds+=list(preds)
        kfold_preds += total_preds  
        
    kfold_preds /= len(model_list)

    write_path = os.path.join(args.output_dir, f"{args.model_name[:-5]}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
        
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(kfold_preds):
            w.write('{},{}\n'.format(id,p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'saint': model = Saint(args)
    if args.model == 'tfbert': model = TfixupBert(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    #1 dataloader #2와 순서를 맞춰주자
    (correct, question, test, tag, time_diff, 
    head, mid, tail, mid_tail, 
    head_answerProb, mid_answerProb, tail_answerProb,
    mask) = batch
    
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    #    saint의 경우 decoder에 들어가는 input이다

    # 패딩을 위해 correct값에 1을 더해준다.
    interaction = correct + 1 
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)

    # set padding index to the first sequence
    interaction_mask[:, 0] = 0 
    interaction = (interaction * interaction_mask).to(torch.int64)

    # print(interaction)
    # exit()

    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    #2 추가 feature
    head_answerProb = ((head_answerProb + 1) * mask).type(torch.FloatTensor)
    mid_answerProb = ((mid_answerProb + 1) * mask).type(torch.FloatTensor)
    tail_answerProb = ((tail_answerProb + 1) * mask).type(torch.FloatTensor)


    time_diff = ((time_diff + 1) * mask).to(torch.int64)
    head = ((head + 1) * mask).to(torch.int64)
    mid = ((mid + 1) * mask).to(torch.int64)
    tail = ((tail + 1) * mask).to(torch.int64)
    mid_tail = ((mid_tail + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    #3 device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)

    time_diff = time_diff.to(args.device)
    head = head.to(args.device)
    mid = mid.to(args.device)
    tail = tail.to(args.device)
    mid_tail = mid_tail.to(args.device)

    head_answerProb = head_answerProb.to(args.device)
    mid_answerProb = mid_answerProb.to(args.device)
    tail_answerProb = tail_answerProb.to(args.device)

    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    #4 
    return (test, question, tag, time_diff, head, mid, tail, mid_tail, 
            head_answerProb, mid_answerProb, tail_answerProb,
            mask, interaction, gather_index, correct)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    print("Loading Model from:", model_path, "...Finished.")
    return model
