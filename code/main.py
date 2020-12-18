import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    modelF = model.Model(args, 'F', checkpoint)
    print(modelF)
    modelU = model.Model(args, 'U', checkpoint)
    print(modelU)
    modelE = model.Model(args, 'E', checkpoint)
    print(modelE)
    modelN = model.Model(args, 'N', checkpoint)
    print(modelN)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, modelF,modelU,modelE,modelN, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

