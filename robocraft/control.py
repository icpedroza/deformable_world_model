import os
import torch
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import taichi as ti
ti.init(arch=ti.cpu)
from plb.engine.taichi_env import TaichiEnv
from plb.config import load

from model import ViT

class ImageBasedPlanner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, img):
        """Convert environment observation to normalized tensor."""
        if isinstance(img, list):  # Handle multi-camera case
            img = img[0]  # Take first camera view
        if isinstance(img, tuple):
            img = img[0]  # Extract RGB if tuple of (rgb, depth)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return self.transform(img).unsqueeze(0).to(self.device)

    def optimize_multi_grip_trajectory(
        self, 
        env, 
        goal_img, 
        n_grips=3, 
        steps_per_grip=40, 
        method="cem",
        n_iterations=100,
        n_samples=64,
        elite_frac=0.1,
        lr=0.01
    ):
        """Optimize multiple gripping trajectories with different initial poses."""
        all_initial_poses = []
        all_actions = []
        current_img = env.render_multi(mode='rgb_array', spp=3)
        
        # Get DINO embeddings of goal
        goal_encoded = self.model.patch_embed(self.preprocess_image(goal_img))

        for grip_idx in range(n_grips):
            print(f"\nOptimizing grip trajectory {grip_idx + 1}/{n_grips}")
            
            if method == "cem":
                poses, actions = self._optimize_grip_cem(
                    env,
                    current_img,
                    goal_encoded,
                    steps_per_grip,
                    n_iterations,
                    n_samples,
                    elite_frac
                )
            else:  # gradient descent
                poses, actions = self._optimize_grip_gd(
                    env,
                    current_img,
                    goal_encoded,
                    steps_per_grip,
                    n_iterations,
                    lr
                )
                
            all_initial_poses.append(poses)
            all_actions.append(actions)
            
            # Execute the optimized trajectory to update environment state
            self.execute_trajectory(env, poses, actions)
            current_img = env.render_multi(mode='rgb_array', spp=3)
            
        return all_initial_poses, all_actions

    def _optimize_grip_cem(
        self,
        env,
        current_img,
        goal_encoded,
        n_steps,
        n_iterations,
        n_samples,
        elite_frac
    ):
        """Optimize single grip using CEM."""
        n_elite = max(1, int(n_samples * elite_frac))
        
        # Initialize distributions for poses and actions
        pose_mean = np.array([0.5, 0.4, 0.5])  # Example initial gripper position
        pose_std = np.array([0.2, 0.1, 0.2])
        
        action_mean = np.zeros((n_steps, 12))  # 12-dim actions from your environment
        action_std = np.ones((n_steps, 12)) * 0.1

        for iteration in range(n_iterations):
            # Sample poses and actions
            pose_samples = np.random.normal(
                pose_mean, pose_std, size=(n_samples, 3)
            )
            action_samples = np.random.normal(
                action_mean, action_std, size=(n_samples, n_steps, 12)
            )
            
            # Evaluate all samples
            rewards = []
            for i in range(n_samples):
                reward = self._evaluate_trajectory(
                    env,
                    pose_samples[i],
                    action_samples[i],
                    current_img,
                    goal_encoded
                )
                rewards.append(reward)
            
            # Select elite samples
            elite_idxs = np.argsort(rewards)[-n_elite:]
            elite_poses = pose_samples[elite_idxs]
            elite_actions = action_samples[elite_idxs]
            
            # Update distributions
            pose_mean = elite_poses.mean(axis=0)
            pose_std = elite_poses.std(axis=0) + 1e-5
            
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-5
            
            if iteration % 10 == 0:
                print(f"CEM Iteration {iteration}, Best Reward: {max(rewards):.4f}")
                
        return pose_mean, action_mean

    def _optimize_grip_gd(
        self,
        env,
        current_img,
        goal_encoded,
        n_steps,
        n_iterations,
        lr
    ):
        """Optimize single grip using gradient descent."""
        # Initialize pose and actions with gradients
        pose = torch.tensor(
            [0.5, 0.4, 0.5], 
            device=self.device, 
            requires_grad=True
        )
        actions = torch.zeros(
            (n_steps, 12), 
            device=self.device, 
            requires_grad=True
        )
        
        optimizer = Adam([pose, actions], lr=lr)
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Forward pass through world model
            current_encoded = self.model.patch_embed(self.preprocess_image(current_img))
            loss = 0
            
            # Simulate trajectory in latent space
            x = current_encoded
            for t in range(n_steps):
                action_t = actions[t].unsqueeze(0)
                x, _ = self.model(x.unsqueeze(1), action_t.unsqueeze(0))
                x = x.squeeze(1)
                
                # Add intermediate losses
                loss += 0.1 * F.mse_loss(x, goal_encoded)
            
            # Final state loss
            loss += F.mse_loss(x, goal_encoded)
            
            # Add pose regularization
            pose_reg = 0.01 * torch.sum(torch.square(pose - torch.tensor([0.5, 0.4, 0.5], device=self.device)))
            loss += pose_reg
            
            loss.backward()
            optimizer.step()
            
            if iteration % 10 == 0:
                print(f"GD Iteration {iteration}, Loss: {loss.item():.4f}")
                
        return pose.detach().cpu().numpy(), actions.detach().cpu().numpy()

    def _evaluate_trajectory(self, env, init_pose, actions, current_img, goal_encoded):
        """Evaluate a trajectory in latent space."""
        # Set initial gripper pose
        env.primitives.primitives[0].set_state(0, [init_pose[0] - 0.2, init_pose[1], init_pose[2], 1, 0, 0, 0])
        env.primitives.primitives[1].set_state(0, [init_pose[0] + 0.2, init_pose[1], init_pose[2], 1, 0, 0, 0])
        
        # Encode current state
        current_encoded = self.model.patch_embed(self.preprocess_image(current_img))
        
        # Roll out trajectory in latent space
        x = current_encoded
        total_reward = 0
        
        for t in range(len(actions)):
            action_t = torch.tensor(actions[t], device=self.device).unsqueeze(0)
            x, _ = self.model(x.unsqueeze(1), action_t.unsqueeze(0))
            x = x.squeeze(1)
            
            # Compute reward as negative distance to goal in latent space
            reward = -F.mse_loss(x, goal_encoded).item()
            total_reward += reward
            
        return total_reward

    def execute_trajectory(self, env, init_pose, actions):
        """Execute optimized trajectory in environment."""
        # Set initial pose
        env.primitives.primitives[0].set_state(0, [init_pose[0] - 0.2, init_pose[1], init_pose[2], 1, 0, 0, 0])
        env.primitives.primitives[1].set_state(0, [init_pose[0] + 0.2, init_pose[1], init_pose[2], 1, 0, 0, 0])
        
        observations = []
        for action in actions:
            env.step(action)
            obs = env.render_multi(mode='rgb_array', spp=3)
            observations.append(obs)
            
        return observations

def main():

    # Set-up environment
    task_name = 'ngrip'
    env_type = '_fixed'

    # gripper_fixed.yml
    cfg = load(f"../simulator/plb/envs/gripper{env_type}.yml") 
    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()

    env.set_state(**state)
    taichi_env = env

    env.renderer.camera_pos[0] = 0.5
    env.renderer.camera_pos[1] = 2.5
    env.renderer.camera_pos[2] = 0.5
    env.renderer.camera_rot = (1.57, 0.0)

    env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
    env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])

    def set_parameters(env: TaichiEnv, yield_stress, E, nu):
        env.simulator.yield_stress.fill(yield_stress)
        _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        env.simulator.mu.fill(_mu)
        env.simulator.lam.fill(_lam)

    set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2


    # Load trained model
    model = ViT(
        image_size=224,
        patch_size=14,
        dim=384,
        depth=6,
        heads=8,
        mlp_dim=2048,
        action_dim=4,
        context_length=5,
        channels=3,
        dropout=0.1
    )
    model.load_state_dict(torch.load('path_to_your_trained_model.pt'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()

    # Initialize planner
    planner = ImageBasedPlanner(model, args, device)

    # Get initial and goal images
    initial_image = env.reset()
    goal_image = env.get_goal_image()  # You'll need to implement this

    # Optimize trajectory
    initial_poses, action_trajectories = planner.optimize_multi_grip_trajectory(
        env=env,
        goal_img=goal_img,
        n_grips=3,
        steps_per_grip=40,
        method="cem"  # or "gd" for gradient descent
    )

    # Execute trajectory
    for pose, actions in zip(initial_poses, action_trajectories):
        observations = planner.execute_trajectory(env, pose, actions)

    # Visualize results (you'll need to implement this based on your needs)
    visualize_trajectory(observations, goal_image)

if __name__ == '__main__':
    main()
