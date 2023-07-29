import torch
import argparse
import random
from utils.data_loader import load_trajs
from utils.metrics import Evaluation_metrics
from models.diffusion import TrajectoryDiffusion
from models.TrajGeneratorNetwork import TrajGeneratorNetwork

def zeroshot_evaluation(args):
    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    traj_diffusion = torch.load('../TrainedModels/trajGDM_' + args.dataset + '.pkl',map_location=device)  # _2333.9665701288186
    traj_diffusion.model_device=device
    traj_diffusion.model.model_device=device
    traj_diffusion.eval()
    dataloader = load_trajs(dataset=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers,train=False)

    if args.dataset=='TDrive':
        num_loc=759
        maxi=27
    elif args.dataset=='Geolife':
        num_loc=12334
        maxi=110
    else:
        raise NotImplementedError(args.dataset)
    print('predicting......')
    correct=0
    total=0
    for x in dataloader:
        x = x[:, :-args.pre_len].to(device)
        y = x[:, -args.pre_len:].to(device)
        predict = traj_diffusion.prediction(x=x, predict_len=args.pre_len, ddim=True, ddim_step=50, ddim_eta=1.0, std=0)
        predict = torch.argmax(predict, dim=-1).to(device)
        for index in range(predict.shape[0]):
            index_y = y[index, :]
            pred_y = predict[index, :]
            if (index_y == pred_y).sum() / args.pre_len > 0.99:
                correct += 1
            total += 1

    print(args.pre_len,' step(s) zero-shot prediction accuracy ',correct/total)
    print('reconstructing......')
    correct = 0
    total = 0
    for x in dataloader:
        location = random.randint(1 + args.pre_len, x.shape[1] - 1)
        x1 = x[:, :location - args.pre_len].to(device)
        x2 = x[:, location:].to(device)
        y = x[:, location:location + args.pre_len].to(device)
        prediction = traj_diffusion.reconstruction(x1, x2, predict_len=args.pre_len, ddim=True ,ddim_step=100,ddim_eta=0., std=0)
        prediction = torch.argmax(prediction, dim=-1)

        for index in range(prediction.shape[0]):
            index_y = y[index, :]
            pred_y = prediction[index, :]
            if (index_y == pred_y).sum() / args.pre_len > 0.99:
                correct += 1
            total += 1

    print(args.pre_len, ' step(s) zero-shot reconstruction accuracy ', correct / total )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--cuda', default="1", type=str)
    parser.add_argument('--dataset', default='TDrive', type=str) #TDrive
    parser.add_argument('--pre_len', default=1, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()
    zeroshot_evaluation(args)