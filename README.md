# Simulating human mobility with a generative trajectory generation model

This is the official Codebase for Simulating Human Mobility with a Generative Trajectory Generation Model.

# Highlights 
We proposed a novel human mobility modeling method named TrajGDM. The method models trajectory generation as a process the uncertainty in the trajectory is gradually removed. A trajectory generator is proposed to predict the uncertainty in a trajectory. Moreover, we defined a trajectory diffusion process and a trajectory generation process to train the trajectory generator and generate a realistic dataset with that. 
We compared the performance of our human mobility modeling method with 5 strong baselines in 2 datasets. Our model achieves great improvement in simulating individual mobility and other metrics while promising the diversity of generated trajectories. Furthermore, by visualizing the trajectory generation process and exploring the latent space of the model, a new perspective on the trajectory generation process is provided.
We conducted zero-shot experiments on two basic trajectory tasks, trajectory prediction and reconstruction. The zero-shot inferring ability of our model verifies the utility of the universal mobility pattern captured through learning the generation process of a trajectory. It also demonstrates the potential of our method in serving as a foundation model in human mobility modeling. 

<img src="https://github.com/chuchen2017/TrajGDM/blob/main/TrajGDM/gifs/1.gif" width="250px"><img src="https://github.com/chuchen2017/TrajGDM/blob/main/TrajGDM/gifs/2.gif" width="250px"><img src="https://github.com/chuchen2017/TrajGDM/blob/main/TrajGDM/gifs/3.gif" width="250px">
<img src="https://github.com/chuchen2017/TrajGDM/blob/main/TrajGDM/gifs/4.gif" width="250px"><img src="https://github.com/chuchen2017/TrajGDM/blob/main/TrajGDM/gifs/5.gif" width="250px"><img src="https://github.com/chuchen2017/TrajGDM/blob/main/TrajGDM/gifs/6.gif" width="250px">



Uncertainty reducing process during trajectory prediction.

# Datasets
Two datasets are used to evaluate the performance of the model. 
## T-Drive: 
The dataset collected the real taxi GPS trajectory in Beijing, China. It contains the trajectory of 10,357 taxis during the period of Feb. 2 to Feb. 8, 2008. The average sample frequency is 2.95 minutes. Considering the data missing problem, we resampled the location of every taxi for every 5 minutes, so all trajectories in the dataset have a fixed time interval. We extracted the positioning points in the six-ring road, which account for 98.2% of all points in the dataset. Then the region in the six-ring road is divided into 27*27 grids by the square with 2000 meters edge length, which was decided by the mobility frequency and averaged moving distance in the dataset. Eventually, there are 169,984 trajectories recorded. 
## Geo-life: 
The dataset was collected from 182 people. The GPS trajectories record their mobility activity over 5 years. We also resampled all trajectories into a 5 minutes time interval and extracted points in six-ring road. Considering the mobility activity is relatively weak, the division is set as 500 meters, so there are 110*110 grids in total. At last, there are 79,360 trajectories left.

We provide coded and cleaned trajectories of two datasets in the datasets file. A dataset is divided into training and testing datasets. The training dataset is used for training the trajectory generating model and evaluate models’ performance in generation. The testing dataset if used for evaluating the performance of zero-shot predicting and reconstructing. 

# Running the experiments
Run runner/trainer.py to train a TrajGDM for T-Drive dataset. 
Run runner/trainer_geolife.py to train a TrajGDM for Geo-Life dataset.
(At the current stage, this part of the code is not included in the current hub, we will release the code as soon as our work is published.)

Run runner/generation_eval.py to generate a synthesized trajectory dataset and evaluate its similarity with the real one using 5 metrics from the paper.

Run runner/zeroshot_eval.py to predict and reconstruct a trajectory with a trained TrajGDM model.

