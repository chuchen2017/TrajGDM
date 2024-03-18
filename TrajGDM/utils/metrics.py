import numpy as np
import torch
from scipy.spatial import distance

class Evaluation_metrics():
    def __init__(self,maxi,num_loc):
        self.ranges=num_loc
        self.maxi=maxi
        self.bins=100

    def js_divergence(self,list1, list2, bins,ranges,density=True):
        p1 = np.histogram(list1, bins=bins, range=(0, ranges), density=density)[0]
        p2 = np.histogram(list2, bins=bins, range=(0, ranges), density=density)[0]
        jsd = distance.jensenshannon(p1, p2)
        return jsd

    def most_common(self,trajs):
        trajs_count = {}
        for batch in range(trajs.shape[0]):
            traj = trajs[batch, :].cpu().tolist()
            trajs_count.setdefault(str(traj), []).append(1)
        trajs_count = dict(map(lambda x: (x[0], sum(x[1])), trajs_count.items()))
        noneone = dict(filter(lambda x: x[1] >= 2, trajs_count.items()))
        num = sum(list(noneone.values()))
        return num

    def distance_array(self,traj):
        distancess = []
        for i in range(traj.shape[1] - 1):
            loc1 = traj[:, i]
            loc2 = traj[:, i + 1]
            x1 = torch.div(loc1, self.maxi, rounding_mode="floor")
            y1 = loc1 % self.maxi
            x2 = torch.div(loc2, self.maxi, rounding_mode="floor")
            y2 = loc2 % self.maxi
            dx = (x1 - x2) ** 2
            dy = (y1 - y2) ** 2
            dis = (dx + dy) ** 0.5
            distancess += dis.view(-1).cpu().tolist()
        return distancess

    def js_traj(self,real_traj, sythsize_traj,  density=True):
        real_o_list = real_traj[:, 0].cpu().tolist()
        real_d_list = real_traj[:, -1].cpu().tolist()
        sythsize_o_list = sythsize_traj[:, 0].cpu().tolist()
        sythsize_d_list = sythsize_traj[:, -1].cpu().tolist()
        real_list = real_traj.view(-1, ).cpu().tolist()
        sythsize_list = sythsize_traj.view(-1, ).cpu().tolist()
        real_dis_list = self.distance_array(real_traj)
        sythsize_dis_list = self.distance_array(sythsize_traj)

        diversity_ratio = self.most_common(trajs=sythsize_traj) / sythsize_traj.shape[0]
        total_jsd = self.js_divergence(real_list, sythsize_list, bins=self.bins, ranges=self.ranges, density=density)
        o_jsd = self.js_divergence(real_o_list, sythsize_o_list, bins=self.bins, ranges=self.ranges, density=density)
        d_jsd = self.js_divergence(real_d_list, sythsize_d_list, bins=self.bins, ranges=self.ranges, density=density)
        dis_jsd = self.js_divergence(real_dis_list, sythsize_dis_list, bins=self.bins, ranges=50, density=density)
        return dis_jsd, total_jsd, o_jsd, d_jsd, diversity_ratio

