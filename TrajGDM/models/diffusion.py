from typing import Optional, List
import numpy as np
import torch

class TrajectoryDiffusion(torch.nn.Module):   #DiffusionSampler
    def __init__(self, model,maxi,lab=2,linear_start: float=0.00085,linear_end: float=0.0120,full_n_steps=1000):
        super().__init__()
        self.model=model
        self.maxi=maxi
        self.lab=lab
        self.linear_start=linear_start
        self.linear_end=linear_end
        self.full_n_steps=full_n_steps

        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, self.full_n_steps, dtype=torch.float64,device=self.model.get_device) ** 2
        self.beta = torch.nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar=alpha_bar
        self.alpha_bar = torch.nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.ddpm_time_steps = np.asarray(list(range(self.full_n_steps)))

        alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])
        self.sqrt_alpha_bar = alpha_bar ** .5
        self.sqrt_1m_alpha_bar = (1. - alpha_bar) ** .5
        self.sqrt_recip_alpha_bar = alpha_bar ** -.5
        self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1) ** .5
        variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
        self.log_var = torch.log(torch.clamp(variance, min=1e-20))
        self.mean_x0_coef = beta * (alpha_bar_prev ** .5) / (1. - alpha_bar)
        self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1. - alpha_bar)

    def diffusion_process(self, x0: torch.Tensor, index: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_alpha_bar[index].view(x0.shape[0],1,1) * x0 + self.sqrt_1m_alpha_bar[index].view(x0.shape[0],1,1) * noise

    def get_uncertainty(self, x: torch.Tensor, t: torch.Tensor):
        e_t=self.model.TrajGenerator(x, t)
        return e_t

    def pred_x0(self, e_t: torch.Tensor, index: torch.Tensor, x: torch.Tensor,):
        sqrt_recip_alpha_bar = self.sqrt_recip_alpha_bar[index].view(x.shape[0], 1,  1)
        sqrt_recip_m1_alpha_bar = self.sqrt_recip_m1_alpha_bar[index].view(x.shape[0], 1,  1)
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t
        return x0

    def generation_training(self, x: torch.Tensor,noise: Optional[torch.Tensor] = None):
        #The code will be released soon
        return 'discrete_loss'

    @torch.no_grad()
    def sampler(self,x,shape):
        x = self.model.LocationEncoder(locs=x,lab=self.lab,maxi=self.maxi)
        x0 = self.model.TrajEncoder(sequence=x)
        noise = torch.randn_like(x0)
        xt = self.sqrt_alpha_bar[self.full_n_steps - 1] * x0 + self.sqrt_1m_alpha_bar[self.full_n_steps - 1] * noise
        xtmean = float(torch.mean(xt))  # .view(-1,self.model.loc_size),dim=0
        xtstd = float(torch.std(xt))  # /1.2 #.view(-1,self.model.loc_size),dim=0
        x_last = torch.normal(xtmean, xtstd, size=shape).to(self.model.get_device)
        return x_last

    @torch.no_grad()
    def sampling(self, x: torch.Tensor, t: torch.Tensor, step: int,temperature=1.0):
        e_t = self.get_uncertainty(x, t)
        if t[0] == 0:
            temperature = 0.
        bs = x.shape[0]
        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1), self.sqrt_recip_alpha_bar[step])
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1), self.sqrt_recip_m1_alpha_bar[step])
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t
        mean_x0_coef = x.new_full((bs, 1, 1), self.mean_x0_coef[step])
        mean_xt_coef = x.new_full((bs, 1, 1), self.mean_xt_coef[step])
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        log_var = x.new_full((bs, 1, 1), self.log_var[step])
        noise = torch.randn(x.shape).to(self.model.get_device)
        noise = noise * temperature
        x_prev = mean + (0.5 * log_var).exp() * noise
        return x_prev, x0, e_t

    @torch.no_grad()
    def sampling_process(self,shape,x_last,temperature: float = 1.):
        x = x_last
        time_steps = np.flip(self.ddpm_time_steps)
        for i, step in zip(range(len(time_steps)), time_steps):
            ts = x.new_full((shape[0],), step, dtype=torch.long,device=self.model.get_device)
            x, pred_x0, e_t = self.sampling(x, ts, step, temperature=temperature)
        return x

    @torch.no_grad()
    def TrajGenerating(self, num_samples=16 ,x:torch.Tensor=None):
        batch_size = num_samples
        shape = [batch_size, self.model.input_len, self.model.loc_size]

        if x != None:
            x_last=self.sampler(x,shape)
        else:
            x_last=torch.randn(shape, device=self.model.get_device)

        x0 = self.sampling_process(shape=shape, temperature=1, x_last=x_last, )  # starts
        trajs = self.model.TrajDecoder(x0)
        return trajs

    @torch.no_grad()
    def ddim_sampling(self, x: torch.Tensor, t: torch.Tensor, index: int, *,temperature: float = 1.,):

        uncertainty = self.get_uncertainty(x, t)

        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        if index == 0:
            temperature = 0

        pred_x0 = (x - sqrt_one_minus_alpha * uncertainty) / (alpha ** 0.5)
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * uncertainty

        if sigma == 0.:
            noise = 0.
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0,uncertainty

    @torch.no_grad()
    def prediction(self, x, predict_len, ddim=True,ddim_step=50,ddim_eta=0.9, std=0):
        locs_embed = self.model.LocationEncoder(locs=x, lab=self.lab, maxi=self.maxi)
        cond = self.model.TrajEncoder(sequence=locs_embed)

        pred = torch.zeros((x.shape[0], predict_len, cond.shape[2]), device=self.model.get_device, dtype=torch.double) + std
        x = torch.cat((cond, pred), dim=1)
        if ddim:
            c = self.full_n_steps // ddim_step
            self.time_steps = np.asarray(list(range(0, self.full_n_steps + 1, c)))
            self.time_steps[-1] = self.time_steps[-1] - 1
            beta = torch.linspace(self.linear_start ** 0.5, self.linear_end ** 0.5, self.full_n_steps,
                                  dtype=torch.float64, device=self.model.get_device) ** 2
            self.beta = torch.nn.Parameter(beta.to(torch.float32), requires_grad=False)
            alpha = 1. - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
            self.alpha_bar = torch.nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])
            self.ddim_sigma = (ddim_eta *((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *(1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

            time_steps = np.flip(self.time_steps)
            for i, step in zip(range(len(time_steps)), time_steps):
                index = len(time_steps) - i - 1
                ts = x.new_full((x.shape[0],), step,dtype=torch.long)
                x, pred_x0, uncertainty = self.ddim_sampling(x=x, t=ts, index=index)
                if cond is not None:
                    cond_t = cond
                    x = torch.cat((cond_t, x[:, -predict_len:, :]), dim=1)

        else:
            time_steps = np.flip(self.ddpm_time_steps)
            for i, step in zip(range(len(time_steps)), time_steps):
                ts = x.new_full((x.shape[0],), step, dtype=torch.long)
                x, pred_x0, e_t = self.sampling(x, ts, step)
                if cond is not None:
                    cond_t = cond
                    x[:, :-predict_len, :] = cond_t

        pres = self.model.TrajDecoder(x[:, -predict_len:, :])
        return pres

    @torch.no_grad()
    def reconstruction(self, x1, x2, predict_len=1, ddim=True,ddim_step=50,ddim_eta=0., std=0):
        cond1 = self.model.LocationEncoder(locs=x1,lab=self.lab,maxi=self.maxi)
        cond1 = self.model.TrajEncoder(sequence=cond1)
        cond2 = self.model.LocationEncoder(locs=x2, lab=self.lab, maxi=self.maxi)
        cond2 = self.model.TrajEncoder(sequence=cond2)

        pred = torch.zeros((x1.shape[0], predict_len, cond2.shape[2]), device=self.model.get_device,dtype=torch.double) + std
        x = torch.cat((cond1, pred, cond2), dim=1)
        if ddim:
            c = self.full_n_steps // ddim_step
            self.time_steps = np.asarray(list(range(0, self.full_n_steps + 1, c)))
            self.time_steps[-1] = self.time_steps[-1] - 1
            beta = torch.linspace(self.linear_start ** 0.5, self.linear_end ** 0.5, self.full_n_steps,
                                  dtype=torch.float64, device=self.model.get_device) ** 2
            self.beta = torch.nn.Parameter(beta.to(torch.float32), requires_grad=False)
            alpha = 1. - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
            self.alpha_bar = torch.nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
            self.ddim_alpha = alpha_bar[self.time_steps].clone().to(torch.float32)
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[self.time_steps[:-1]]])
            self.ddim_sigma = (ddim_eta * ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) * (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

            time_steps = np.flip(self.time_steps)
            for i, step in zip(range(len(time_steps)), time_steps):
                index = len(time_steps) - i - 1
                ts = x.new_full((x.shape[0],), step, dtype=torch.long)
                x, pred_x0, uncertainty = self.ddim_sampling(x=x, t=ts, index=index)
                if cond1 is not None:
                    x = x[:, cond1.shape[1]:cond1.shape[1] + predict_len, :]
                    x = torch.cat((cond1, x, cond2), dim=1)

        else:
            time_steps = np.flip(self.ddpm_time_steps)
            for i, step in zip(range(len(time_steps)), time_steps):
                ts = x.new_full((x.shape[0],), step, dtype=torch.long)
                x, pred_x0, e_t = self.sampling(x, ts, step)
                if cond1 is not None:
                    x = x[:, cond1.shape[1]:cond1.shape[1] + predict_len, :]
                    x = torch.cat((cond1, x, cond2), dim=1)

        pres = self.model.TrajDecoder(x[:, cond1.shape[1]:cond1.shape[1] + predict_len, :])
        return pres