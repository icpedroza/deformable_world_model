import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import cv2
import datetime

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import ViT

class RobotImageDataset(Dataset):
    def __init__(self, data_dir, context_length):
        self.data_dir = data_dir
        self.context_length = context_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Get all trajectory directories
        self.traj_dirs = []
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
        for vid_idx in range(1000):  # Large number, will break when no more exist
            traj_path = os.path.join(data_dir, f"{vid_idx:03d}")
            if not os.path.exists(traj_path):
                break
            # print(f"Found trajectory directory: {traj_path}")
            self.traj_dirs.append(traj_path)

        if len(self.traj_dirs) == 0:
            raise ValueError(f"No trajectory directories found in {data_dir}")
            
        # Get number of timesteps per trajectory
        sample_path = os.path.join(self.traj_dirs[0], "000_rgb_0.png")
        if not os.path.exists(sample_path):
            raise ValueError(f"No images found at {sample_path}")
        max_steps = 0
        while os.path.exists(os.path.join(self.traj_dirs[0], f"{max_steps:03d}_rgb_0.png")):
            max_steps += 1

        self.timesteps_per_traj = max_steps
        
    def __len__(self):
        return len(self.traj_dirs) * (self.timesteps_per_traj - self.context_length)
        
    def __getitem__(self, idx):
        # Convert idx to trajectory and timestep
        traj_idx = idx // (self.timesteps_per_traj - self.context_length)
        start_step = idx % (self.timesteps_per_traj - self.context_length)
        
        # Load sequence of images and actions
        states = []
        actions = []
        traj_dir = self.traj_dirs[traj_idx]
        
        for t in range(start_step, start_step + self.context_length + 1):  # +1 for target frame
            # Load RGB image (using first camera view)
            img_path = os.path.join(traj_dir, f"{t:03d}_rgb_0.png")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = self.transform(img)
            states.append(img)
            
            if t < start_step + self.context_length:  # Don't need action for final frame
                # Load action
                action_path = os.path.join(traj_dir, f"{t:03d}_action.npy")
                action_dict = np.load(action_path, allow_pickle=True).item()
                # Combine midpoint and angle into single tensor
                midpoint = torch.from_numpy(action_dict["midpoint"]).float()
                angle = torch.tensor([action_dict["cur_angle"]]).float()
                action = torch.cat([midpoint, angle])  # 4-dimensional: [x,y,z,angle]
                actions.append(action)
        
        states = torch.stack(states)  # (context_length+1, C, H, W)
        actions = torch.stack(actions)  # (context_length, action_dim)
        
        return {
            'states': states[:-1],  # Input frames
            'actions': actions,
            'targets': states[1:]    # Target frames
        }

def train_world_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    learning_rate=1e-4,
    save_dir='checkpoints'
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Create a unique subfolder based on current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_checkpoint_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    epochs = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            states = batch['states'].to(device)    # B, H, C, h, w
            actions = batch['actions'].to(device)  # B, H, action_dim
            targets = batch['targets'].to(device)    # B H C H W  (H future states)
            
            optimizer.zero_grad()
            
            # Get predicted next state in DINO embedding space
            pred_embeddings = model(states, actions)  # (B, N, D)
            
            # Get target embedding
            with torch.no_grad():
                target_embeddings = []
                for i in range(targets.shape[1]):
                    target_embed = model.patch_embed(targets[:, i])
                    target_embeddings.append(target_embed)
                target_embeddings = torch.stack(target_embeddings, dim=1)  # B H N D
            
            # Compute loss in embedding space
            loss = F.mse_loss(pred_embeddings, target_embeddings)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                targets = batch['targets'].to(device)
                
                pred_embeddings = model(states, actions)
                target_embeddings = []
                for i in range(targets.shape[1]):
                    target_embed = model.patch_embed(targets[:, i])
                    target_embeddings.append(target_embed)
                target_embeddings = torch.stack(target_embeddings, dim=1)
                
                loss = F.mse_loss(pred_embeddings, target_embeddings)
                val_loss += loss.item()
                
        val_loss /= len(val_dataloader)

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch + 1)
        
        print(f"Epoch {epoch+1}: train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_checkpoint_dir, 'best_model.pt'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(run_checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        # Save loss plot every epoch
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Using log scale since losses can vary widely
        
        # Save plot
        plt.savefig(os.path.join(run_checkpoint_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save losses as numpy arrays for later analysis
        np.save(os.path.join(run_checkpoint_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(run_checkpoint_dir, 'val_losses.npy'), np.array(val_losses))

    # Create and save final plots with more detail
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses (Linear Scale)')
    plt.legend()
    plt.grid(True)
    
    # Log scale plot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Losses (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_checkpoint_dir, 'final_loss_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Hyperparameters
    context_length = 5
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    image_size = 224
    patch_size = 14
    embed_dim = 384  # DINOv2 embedding dimension
    action_dim = 4  # From your gripper actions
    dino = False
    
    # Initialize model
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        dim=embed_dim,
        depth=6,
        heads=8,
        mlp_dim=2048,
        action_dim=action_dim,
        context_length=context_length,
        channels=3,
        dropout=0.1,
        dino=dino
    )
    
    if dino:
        # Load and freeze DINO weights
        dinov2 = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')
        model.patch_embed.proj.weight.data = dinov2.patch_embed.proj.weight.data
        model.patch_embed.proj.bias.data = dinov2.patch_embed.proj.bias.data
        
        # Freeze patch embedding
        for param in model.patch_embed.parameters():
            param.requires_grad = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_dataset = RobotImageDataset(
        data_dir='/home/ianpedroza/RoboCraft/simulator/dataset/ngrip_fixed_training',
        context_length=context_length
    )
    
    val_dataset = RobotImageDataset(
        data_dir='/home/ianpedroza/RoboCraft/simulator/dataset/ngrip_fixed_validation',
        context_length=context_length
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Train model
    train_world_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        device=device,
        learning_rate=learning_rate,
        save_dir='checkpoints'
    )

if __name__ == '__main__':
    main()