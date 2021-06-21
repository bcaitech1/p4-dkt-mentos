from torch.optim import Adam, AdamW
from adamp import AdamP


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == 'adamp':
        # define your params
        optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()
    
    return optimizer