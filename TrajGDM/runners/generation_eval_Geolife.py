import torch
import argparse
from utils.data_loader import load_trajs
from utils.metrics import Evaluation_metrics
from models.diffusion import TrajectoryDiffusion
from models.TrajGeneratorNetwork import TrajGeneratorNetwork

def generation_evaluation(args):
    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    traj_diffusion = torch.load('../TrainedModels/trajGDM_' + args.dataset + '.pkl',map_location=device)  # _1886.1173384562853  3438.2558547773724
    traj_diffusion.model_device=device
    traj_diffusion.model.model_device=device
    traj_diffusion.eval()
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

    trajs_list = []
    for x in dataloader:
        trajs_list.append(x.cpu())
    total_x = torch.cat(trajs_list, dim=0)

    print('Generating Trajectories......')
    if args.according_real:
        trajs_generated = []
        for x in dataloader:
            trajs = traj_diffusion.TrajGenerating(x=x.to(device), num_samples=args.batch_size)
            trajs = torch.argmax(trajs, dim=-1, keepdim=False)
            trajs_generated.append(trajs)
            generated = torch.cat(trajs_generated, dim=0)
            dis_jsd, total_jsd, o_jsd, d_jsd, ratio = metrics.js_traj(real_traj=total_x, sythsize_traj=generated)
            print('Number of Generated Trajectories:', int(generated.shape[0]), ' JSD Moving: ', dis_jsd,
                  ' Distribution ', total_jsd, ' O-Dis ', o_jsd, ' D-Dis ', d_jsd, ' Diversity ', ratio)
    else:
        trajs_generated = []
        for i in range(args.num_evaluation // args.batch_size):
            trajs = traj_diffusion.TrajGenerating(num_samples=args.batch_size, x=None)
            trajs = torch.argmax(trajs, dim=-1, keepdim=False)
            trajs_generated.append(trajs)
            generated = torch.cat(trajs_generated, dim=0)
            dis_jsd, total_jsd, o_jsd, d_jsd, ratio = metrics.js_traj(real_traj=total_x, sythsize_traj=generated)
            print('Number of Generated Trajectories:', int(generated.shape[0]), ' JSD Moving: ', dis_jsd,
                  ' Distribution ', total_jsd, ' O-Dis ', o_jsd, ' D-Dis ', d_jsd, ' Diversity ', ratio)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--cuda', default="1", type=str)
    parser.add_argument('--dataset', default='Geolife', type=str) #TDrive  Geolife
    parser.add_argument('--length', default=12, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--num_evaluation', default=6000, type=int)
    parser.add_argument('--according_real', default=True, type=bool)
    args = parser.parse_args()
    generation_evaluation(args)