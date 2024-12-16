import os
import torch
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.spatial.distance import cdist
import datetime

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
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, img):
        """Convert environment observation to normalized tensor."""
        if isinstance(img, list):
            img = img[0]
        if isinstance(img, tuple):
            img = img[0]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return self.transform(img).unsqueeze(0).to(self.device)

    def action_to_gripper_states(self, midpoint, angle):
        """Convert midpoint and angle to gripper states."""
        direction = np.array([
            np.cos(angle),
            0,  # y component is always 0 since movement is horizontal
            np.sin(angle)
        ])
        direction = direction / np.linalg.norm(direction)
        
        # Set grippers on either side of midpoint
        gripper_offset = 0.2  # Distance from midpoint to each gripper
        gripper1_pos = midpoint - direction * gripper_offset
        gripper2_pos = midpoint + direction * gripper_offset
        
        # Add quaternion orientation (1,0,0,0 for upright)
        gripper1_state = np.concatenate([gripper1_pos, [1, 0, 0, 0]])
        gripper2_state = np.concatenate([gripper2_pos, [1, 0, 0, 0]])
        
        return gripper1_state, gripper2_state

    def action_to_raw_commands(self, midpoint, angle, rate=0.12):
        """Convert midpoint and angle to raw action commands."""
        direction = np.array([
            np.cos(angle),
            0,
            np.sin(angle)
        ])
        direction = direction / np.linalg.norm(direction)
        
        # Create action for both grippers moving in opposite directions
        prim1_action = rate * direction
        zero_pad = np.array([0, 0, 0])
        raw_action = np.concatenate([prim1_action, zero_pad, -prim1_action, zero_pad])
        
        return raw_action

    def optimize_multi_grip_trajectory(
        self, 
        env, 
        goal_img, 
        goal_points,
        n_grips=3, 
        steps_per_grip=40,
        n_iterations=100,
        n_samples=64,
        elite_frac=0.1,
        output_dir='rollouts'
    ):
        """Optimize multiple gripping trajectories."""
        os.makedirs(output_dir, exist_ok=True)
        all_actions = []
        current_img = env.render(mode='rgb_array', spp=3)[0]
        with torch.no_grad():
            goal_encoded = self.model.patch_embed(self.preprocess_image(goal_img))
        
        for grip_idx in range(n_grips):
            print(f"\nOptimizing grip trajectory {grip_idx + 1}/{n_grips}")

            # move grippers off screen
            prim1, prim2, cur_angle = random_pose()
            test_prim = prim1 * 10
            update_primitive(env, test_prim, test_prim)
            current_img = env.render(mode='rgb_array', spp=3)[0]
            
            # Optimize single action that will be repeated
            optimized_action = self._optimize_grip_cem(
                env,
                current_img,
                goal_encoded,
                steps_per_grip,
                n_iterations,
                n_samples,
                elite_frac
            )
            all_actions.append(optimized_action)
            
            # Execute the optimized trajectory
            observations = self.execute_trajectory(
                env, 
                optimized_action, 
                steps_per_grip,
                f"{output_dir}/{grip_idx:03d}"
            )
            current_img = observations[-1]
            
            # Save video of rollout
            self.save_video(observations, f"{output_dir}/{grip_idx:03d}/vid{grip_idx:03d}.mp4")
            
        prim1, prim2, cur_angle = random_pose()
        test_prim = prim1 * 10
        update_primitive(env, test_prim, test_prim)
        # Calculate final image loss
        final_img = current_img
        image_loss = F.mse_loss(
            self.preprocess_image(final_img),
            self.preprocess_image(goal_img)
        ).item()
        print(f"Final image loss: {image_loss:.6f}")

        # Get current observation points
        obs = get_obs(env, 300)
        final_points = obs[0][:300]
        
        # Calculate Chamfer distance between final and goal points
        chamfer_dist = self.compute_chamfer_distance(final_points, goal_points)
        print(f"Final Chamfer distance: {chamfer_dist:.6f}")
        
        # Calculate Chamfer distance if particle positions available
        # if hasattr(env, 'get_particles'):
        #     final_particles = env.get_particles()
        #     goal_particles = env.get_goal_particles()  # You'll need to implement this
        #     chamfer_dist = self.compute_chamfer_distance(final_particles, goal_particles)
        #     print(f"Final Chamfer distance: {chamfer_dist:.6f}")
        
        return all_actions

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
        """Optimize single grip action using CEM."""
        n_elite = max(1, int(n_samples * elite_frac))
        
        # Initialize distributions for action (midpoint and angle)
        action_mean = np.array([0.5, 0.4, 0.5, 0.0])  # [x,y,z,angle]
        action_std = np.array([0.2, 0.1, 0.2, np.pi/4])
        
        for iteration in range(n_iterations):
            # Sample actions
            action_samples = np.random.normal(
                action_mean, action_std, size=(n_samples, 4)
            )
            
            # Evaluate all samples
            rewards = []
            for i in range(n_samples):
                reward = self._evaluate_trajectory(
                    env,
                    action_samples[i],
                    n_steps,
                    current_img,
                    goal_encoded
                )
                rewards.append(reward)
            
            # Select elite samples
            elite_idxs = np.argsort(rewards)[-n_elite:]
            elite_actions = action_samples[elite_idxs]
            
            # Update distributions
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-5
            
            if iteration % 5 == 0:
                print(f"CEM Iteration {iteration}, Best Reward: {max(rewards):.4f}")
                
        return action_mean

    def _evaluate_trajectory(self, env, action, n_steps, current_img, goal_encoded):
        """Evaluate a trajectory in latent space."""
        CONTEXT_LENGTH = 5
        with torch.no_grad():
            current_state = self.preprocess_image(current_img)  # (1, C, H, W)
            current_embedding = self.model.patch_embed(current_state)  # (1, N, D)
            
            action_tensor = torch.tensor(action, device=self.device, dtype=torch.float32)
            action_tensor = action_tensor.unsqueeze(0).unsqueeze(0)

            state_embeddings = [current_embedding] * CONTEXT_LENGTH
            
            # For each step, predict next state using previous state
            for step in range(n_steps):
                # Create input sequence with current state
                embedding_sequence = torch.stack(state_embeddings[-CONTEXT_LENGTH:], dim=1)  # (1, 5, N, D)

                # Create matching action sequence
                action_sequence = action_tensor.repeat(1, CONTEXT_LENGTH, 1)  # (1, step+1, action_dim)
                
                # Forward pass through transformer part only
                predicted_states = self.model.transformer(embedding_sequence + 
                                                        self.model.action_encoder(action_sequence).unsqueeze(2).expand(-1, -1, self.model.patch_embed.num_patches, -1))
                
                # Extract predicted next embedding and add to sequence
                next_embedding = predicted_states[:, -1]  # (1, N, D)
                state_embeddings.append(next_embedding)

                if len(state_embeddings) > CONTEXT_LENGTH + 1:
                    state_embeddings.pop(0)
            
            reward = -F.mse_loss(state_embeddings[-1], goal_encoded).item()

        return reward

    def execute_trajectory(self, env, action, n_steps, save_dir):
        """Execute optimized trajectory in environment."""
        os.makedirs(save_dir, exist_ok=True)
        observations = []

        midpoint = np.array([0.5, 0.14, 0.5, 0, 0, 0])
        
        # Set initial gripper states
        gripper1_state, gripper2_state = self.action_to_gripper_states(midpoint[:3], action[3])
        env.primitives.primitives[0].set_state(0, gripper1_state)
        env.primitives.primitives[1].set_state(0, gripper2_state)
        
        # Convert to raw action commands
        raw_action = self.action_to_raw_commands(midpoint[:3], action[3])
        
        # Execute steps
        for step in range(n_steps):
            env.step(raw_action)
            obs = env.render(mode='rgb_array', spp=3)[0]
            observations.append(obs)
            
            # Save image
            cv2.imwrite(f"{save_dir}/{step:03d}_rgb_0.png", obs[..., ::-1])
        
        return observations

    def save_video(self, frames, output_path):
        """Save frames as video."""
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            25,  # fps
            (width, height)
        )
        
        for frame in frames:
            writer.write(frame[..., ::-1])  # Convert RGB to BGR
        writer.release()

    def compute_chamfer_distance(self, points1, points2):
        """Compute Chamfer distance between two point clouds."""
        # Compute pairwise distances
        distances1 = cdist(points1, points2)
        distances2 = cdist(points2, points1)
        
        # Compute minimum distances in both directions
        chamfer1 = np.mean(np.min(distances1, axis=1))
        chamfer2 = np.mean(np.min(distances2, axis=1))
        
        # Return symmetric Chamfer distance
        return (chamfer1 + chamfer2) / 2

def random_pose(radius=0.4, p_noise_scale=0.01, midpoint=np.array([0.5, 0.14, 0.5, 0, 0, 0])):
    p_noise_x = p_noise_scale * (np.random.randn() * 2 - 1)
    p_noise_z = p_noise_scale * (np.random.randn() * 2 - 1)

    p_noise = np.clip(np.array([p_noise_x, 0, p_noise_z]), a_min=-0.1, a_max=0.1)
    
    new_mid_point = midpoint[:3] + p_noise

    rot_noise = np.random.uniform(0, np.pi)

    x1 = new_mid_point[0] - radius * np.cos(rot_noise)
    z1 = new_mid_point[2] + radius * np.sin(rot_noise)
    x2 = new_mid_point[0] + radius * np.cos(rot_noise)
    z2 = new_mid_point[2] - radius * np.sin(rot_noise)
    y = new_mid_point[1]
    z_vec = np.array([np.cos(rot_noise), 0, np.sin(rot_noise)])

    gripper1_pos = np.array([x1, y, z1])
    gripper2_pos = np.array([x2, y, z2])
    quat = np.array([1, 0, 0, 0])
    
    direction = new_mid_point - gripper1_pos
    rotation = np.arctan2(direction[2], direction[0])

    return np.concatenate([gripper1_pos, quat]), np.concatenate([gripper2_pos, quat]), rotation

def update_primitive(env, prim1_list, prim2_list):
    env.primitives.primitives[0].set_state(0, prim1_list)
    env.primitives.primitives[1].set_state(0, prim2_list)

def get_obs(env, n_particles, t=0):
    x = env.simulator.get_x(t)
    v = env.simulator.get_v(t)
    step_size = len(x) // n_particles
    return x[::step_size], v[::step_size]

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
    checkpoint_path = '/home/ianpedroza/RoboCraft/robocraft/checkpoints/2024-12-15_18-08-32/best_model.pt'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize planner
    planner = ImageBasedPlanner(model, device)

    for i in range(1, 3):
        goal_img_bgr = cv2.imread(f'/home/ianpedroza/RoboCraft/robocraft/goal_states/{i}/goal_state.png')
        goal_img = cv2.cvtColor(goal_img_bgr, cv2.COLOR_BGR2RGB)

        goal_points = np.load(f'/home/ianpedroza/RoboCraft/robocraft/goal_states/{i}/gtp.npy')

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"planning_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save goal image
        cv2.imwrite(os.path.join(output_dir, 'goal_image.png'), goal_img[..., ::-1])

        # move grippers off screen
        prim1, prim2, cur_angle = random_pose()
        test_prim = prim1 * 10
        update_primitive(env, test_prim, test_prim)

        initial_img = env.render(mode='rgb_array', spp=3)[0]
        cv2.imwrite(os.path.join(output_dir, 'initial_image.png'), initial_img[..., ::-1])

        # Optimize trajectory
        action_trajectories = planner.optimize_multi_grip_trajectory(
            env=env,
            goal_img=goal_img,
            goal_points=goal_points,
            n_grips=3,
            steps_per_grip=40,
            n_iterations=2,
            n_samples=2,
            elite_frac=0.1,
            output_dir=output_dir
        )

if __name__ == '__main__':
    main()
