import time
import argparse
import numpy as np
import torch
from utils.data_loader import load_trajs
from utils.metrics import Evaluation_metrics
from models.diffusion import TrajectoryDiffusion
from models.TrajGeneratorNetwork import TrajGeneratorNetwork

def trainer(args):
    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    dataloader = load_trajs(dataset=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers)

    if args.dataset=='TDrive':
        num_loc=759
        maxi=27
    elif args.dataset=='Geolife':
        num_loc=12334
        maxi=110
    else:
        raise NotImplementedError(args.dataset)

    metrics=Evaluation_metrics(num_loc=num_loc,maxi=maxi)

    trajgenerator = TrajGeneratorNetwork(num_location=num_loc, location_embedding=args.num_hidden,maxi=maxi,num_head=args.TrajGenerator_heads,
                                         lstm_hidden=args.num_hidden, device=device,TrajGenerator_Translayers=args.TrajGenerator_Translayers,TrajGenerator_LSTMlayers=args.TrajGenerator_LSTMlayers,
                                         input_len=args.length).double().to(device)

    traj_diffusion = TrajectoryDiffusion(model=trajgenerator,maxi=maxi,lab=2,linear_start=0.00085, linear_end=0.0120,full_n_steps=1000).double().to(device)
    optimizer = torch.optim.Adam(trajgenerator.parameters(), lr=0.0001)  # 0
    trajgenerator.train()
    print('Start Training......')
    for e in range(args.epoch):
        trajgenerator.train()
        traj_diffusion.train()
        train_loss = []
        trajs_list = []
        t1 = time.time()
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            loss =  traj_diffusion.generation_training(x)
            loss.backward()
            optimizer.step()
            train_loss.append(float(loss.cpu().detach()))
            trajs_list.append(x.cpu())
            #break
        t2 = time.time()
        print('epoch ', e, ' minutes ', round((t2 - t1) / 60), ' training loss ', np.mean(train_loss), flush=True)

        if e != 0 and e % args.eval_epoch == 0:
            with torch.no_grad():
                trajgenerator.eval()
                traj_diffusion.eval()
                trajs_generated=[]
                for i in range(args.num_evaluation//args.batch_size):
                    trajs = traj_diffusion.TrajGenerating(num_samples=args.batch_size,x=None)
                    trajs = torch.argmax(trajs, dim=-1,keepdim=False)
                    trajs_generated.append(trajs)

                trajs_generated=torch.cat(trajs_generated,dim=0)
                total_x = torch.cat(trajs_list, dim=0)
                dis_jsd,total_jsd, o_jsd, d_jsd,  ratio = metrics.js_traj(real_traj=total_x, sythsize_traj=trajs_generated)
                print('JSD', dis_jsd,total_jsd, o_jsd, d_jsd,  ratio)

            torch.save(traj_diffusion,'../TrainedModels/trajGDM_' + args.dataset + '_' + str(float(np.mean(train_loss))) + '.pkl')
        #break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--cuda', default="0", type=str)
    parser.add_argument('--dataset', default='Geolife', type=str)  #Geolife  TDrive
    parser.add_argument('--batch_size', default=512+128, type=int) #+128
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--num_hidden', default=512, type=int)
    parser.add_argument('--TrajGenerator_Translayers', default=2, type=int)
    parser.add_argument('--TrajGenerator_heads', default=2, type=int)
    parser.add_argument('--TrajGenerator_LSTMlayers', default=3, type=int)
    parser.add_argument('--num_evaluation', default=512, type=int)
    parser.add_argument('--eval_epoch', default=10, type=int)
    parser.add_argument('--length', default=12, type=int)
    args = parser.parse_args()
    trainer(args)