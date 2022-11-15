import  torch, os
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from    dataset.miniimagenet import MiniImagenet
from    maml import MAML

import wandb
from tqdm import tqdm
import torch.distributed as dist

def distributedGPU(dataset, model, local_rank=0, world_size=0):
    

    # Set the rank for each processes
    if local_rank == 0:
        local_rank = int(os.environ["LOCAL_RANK"])

    # (Option) Assign GPUs - it is more encouraged to use `CUDA_VISIBLE_DEVICES` environment variable
    torch.cuda.set_device(local_rank)

    # Get the total number of GPUs to be used - used for distributing data
    if world_size == 0:
        world_size = torch.distributed.get_world_size() 

    # Split the data with respect to the world_size (total number of GPUs)
    # Each GPU will be assign divided portion of dataset -> Q, Then does each GPU will get the same training indices?
    # If the dataset is not evenly divisible by the world_size, then you can drop the remainder by setting `drop_last=True`, which is default
    # Otherwise, `drop_last=False` will get extra portions of data by generating similar batch_sized data
    dataset = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, drop_last=False)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
    
    return dataset, model


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # Initialize Process Groups
    # dist.init_process_group(backend='nccl')
    wandb.init()
    # wandb.init(project="project-name", reinit=True)
    wandb.run.name = wandb.run.id
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    wandb.config.update(args)
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    # local_rank = int(os.environ["LOCAL_RANK"])

    # (Option) Assign GPUs - it is more encouraged to use `CUDA_VISIBLE_DEVICES` environment variable
    # torch.cuda.set_device(local_rank)

    device = torch.device('cuda')
    # device = torch.device(local_rank)

    maml = MAML(args, config).to(device)
    model = torch.nn.DataParallel(maml,
                                                  device_ids=[0,1],
                                                  output_device=0)
    wandb.watch(maml)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('/solee/miniImagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batch_size=10000)
    mini_test = MiniImagenet('/solee/miniImagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batch_size=100)
    # mini, maml = distributedGPU(mini, maml)
    # mini_test, maml = distributedGPU(mini_test, maml)


    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm(db)):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)
            metrics = {"train/train_acc": accs, 
                        "train/train_acc_mean": accs.mean(axis=0), 
                       "train/epoch": (step + 1 + (len(db) * epoch)) / len(db)}

            if step % 30 == 0:
                # print('step:', step, '\ttraining acc:', accs)
                wandb.log(metrics)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=8, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                # print('Test acc:', accs)
                metrics_test = {"test/test_acc": accs}
                wandb.log({**metrics, **metrics_test})

    wandb.finish()



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument("--local_rank", type=int, default=0)

    args = argparser.parse_args()

    main()