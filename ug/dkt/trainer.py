import os
import torch
import numpy as np
from glob import glob


from .dataloader import get_loaders
from .optimizer import get_optimizer, get_lr
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, Transformer, Saint, LastQuery

import wandb

def run(args, train_data, valid_data):
    print(args)

    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    
    # model save path 설정
    model_dir = os.path.join(args.model_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)
    save_name = None
    model_number = glob(os.path.join(model_dir, "*"))
    if len(model_number) == 0:
        save_name = "model_0.pt"
    else:
        # model_number = [int(m.split(".")[0][-1]) for m in model_number]
        model_number = [int(m.split(".")[0].split("_")[1]) for m in model_number]
        model_number.sort()
        save_name = "model_" + str(model_number[-1] + 1) + ".pt"
    print("[saved model path] ", args.model, "/", save_name)
    
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10
            
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    # print("optimizer lr 2: ", get_lr(optimizer))
    scheduler = get_scheduler(optimizer, args)

    wandb.watch(model)

    # print("optimizer lr 3: ", get_lr(optimizer))

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)
        
        ### VALID
        auc, acc,_ , _ = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc, "learning_rate": get_lr(optimizer)})
        if auc > best_auc:
            best_auc = auc

            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                model_dir, save_name,
            )
            early_stopping_counter = 0

            print(f"Save best AUC model! [{auc}]")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()


def train(train_loader, model, optimizer, args):
    # scheduler = get_scheduler(optimizer, args)

    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        inputs = process_batch(batch, args)
        preds = model(inputs)
        targets = inputs[3] # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        # print("learning rate: ", get_lr(optimizer))

        # scheduler
        # if args.scheduler == 'plateau':
        #     scheduler.step(best_auc)
        # else:
        #     scheduler.step()

        # wandb.log({"lr": get_lr(optimizer)})

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
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        inputs = process_batch(batch, args)

        preds = model(inputs)
        targets = inputs[3] # correct


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
        input = process_batch(batch, args)

        preds = model(input)
        

        # predictions
        preds = preds[:,-1]
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, args.output_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))




def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'transformer' : model = Transformer(args)
    if args.model == 'saint' : model = Saint(args)
    if args.model == 'query' : model = LastQuery(args)
    

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    # test, question, tag, correct, mask = batch
    correct, question, test, tag, paperid, head, mid, tail, tail_prob, test_split_prob, time_diff, mask = batch
    
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    # #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    # #    saint의 경우 decoder에 들어가는 input이다
    # interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    # interaction = interaction.roll(shifts=1, dims=1)
    # interaction[:, 0] = 0 # set padding index to the first sequence
    # interaction = (interaction * mask).to(torch.int64)

    # print(interaction)
    # exit()
    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # 추가된 feature
    paperid = ((paperid + 1) * mask).to(torch.int64)
    head = ((head + 1) * mask).to(torch.int64)
    mid = ((mid + 1) * mask).to(torch.int64)
    tail = ((tail + 1) * mask).to(torch.int64)
    # tail_prob = ((tail_prob + 1) * mask).to(torch.int64)
    # time = ((time + 1) * mask).to(torch.int64)

    # numeric feature -> float
    tail_prob = ((tail_prob + 1) * mask).to(torch.float32)
    test_split_prob = ((test_split_prob + 1) * mask).to(torch.float32)
    time_diff = ((time_diff + 1) * mask).to(torch.float32)
    # time = ((time + 1) * mask).to(torch.float32)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    paperid = paperid.to(args.device)
    head = head.to(args.device)
    mid = mid.to(args.device)
    tail = tail.to(args.device)
    tail_prob = tail_prob.to(args.device)
    test_split_prob = test_split_prob.to(args.device)
    time_diff = time_diff.to(args.device)
    # time = time.to(args.device)

    return (test, question,
            tag, correct, mask,
            interaction, paperid, head, mid, tail, tail_prob, test_split_prob, time_diff, gather_index)


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
    
    
    model_path = os.path.join(args.model_dir, args.model, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    
    print("Loading Model from:", model_path, "...Finished.")
    return model