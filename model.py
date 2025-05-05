import torch
import copy
from collections import deque
import random
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        # TODO: Initialize the buffer
        self.data = deque(maxlen = capacity)

    # TODO: Implement the add method
    # def add(self, ...):
    def add(self, state,action, new_state,reward,done):
        self.data.append((copy.deepcopy(state),action,copy.deepcopy(new_state),reward,done))
    # TODO: Implement the sample method
    # def sample(self, ...):
    def sample(self,sample_size):
        return random.sample(self.data,min(sample_size,len(self.data)))

# for windows cuda

# Check if CUDA is available
print(torch.cuda.is_available())  # Should return True

# Get the current CUDA device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)
class CNN_DQN(torch.nn.Module):
    def __init__(self, state_shape, n_actions):
        super(CNN_DQN, self).__init__()
        hidden_dim = 512
        """self.cnn = torch.nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, stride=4, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),  # 輸出為 (batch_size, 2592)
            nn.Linear(2592, hidden_dim),
            nn.ReLU(),
        )
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, stride=4, kernel_size=8),  # (batch_size, 20, 20, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=4),  # (batch_size, 9, 9, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3),  # (batch_size, 7, 7, 64)
            nn.ReLU(),
            nn.Flatten(),  # (batch_size, 3136)
        )

        self.fc_A = nn.Sequential(
            nn.Linear(3136,hidden_dim),
            nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_actions)
        )
        self.fc_V = nn.Sequential(
            nn.Linear(3136,hidden_dim),
            nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, img):
        y_cnn = self.cnn(img)
        A = self.fc_A(y_cnn)
        V = self.fc_V(y_cnn)
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size, device):
        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.device = device

        self.q_network = CNN_DQN(state_size, action_size).to(self.device)
        self.target_network = CNN_DQN(state_size, action_size).to(self.device)
        self.lr = 1e-4
        self.epsilon_min = 0.01
        self.epsilon_decay_steps = 250000
        self.optimizer = optim.Adam(self.q_network.parameters(), self.lr)
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = 0.63
        self.step_count = 0
        self.loss_fn = nn.MSELoss()
        self.epsilon = 1
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = 32
        self.update_frequency = 4 # Update Q-network every 4 steps
        self.target_update_frequency = 1000  # Update target network every 1000 steps
        for param in self.target_network.parameters():
            param.requires_grad = False
        
    def load_model(self,path_q,path_target):
        checkpoint_q = torch.load(path_q,map_location=self.device)
        checkpoint_target = torch.load(path_target,map_location=self.device)
        self.q_network.load_state_dict(checkpoint_q)
        self.target_network.load_state_dict(checkpoint_target)
        
    def save_checkpoint(self, checkpoint_dir="checkpoints", filename_prefix="mario_dqn",save_buffer=True):
        """
        Save model parameters, optimizer state, training progress and replay buffer
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save models and optimizer
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'gamma': self.gamma,
            'lr': self.lr,
        }
        
        # Save the main checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"{filename_prefix}_checkpoint.pth"))
        
        print(f"Checkpoint saved to {checkpoint_dir}/{filename_prefix}_checkpoint.pth")
        if save_buffer:
            try:
                with open(os.path.join(checkpoint_dir, f"{filename_prefix}_replay_buffer.pkl"), 'wb') as f:
                    pickle.dump(self.replay_buffer.data, f)
                print(f"Replay buffer saved with {len(self.replay_buffer.data)} experiences")
    
            except:
                print(f"Replay buffer saved with {len(self.replay_buffer.data)} experiences")
    
    def load_checkpoint(self, checkpoint_dir="checkpoints", filename_prefix="mario_dqn", load_replay_buffer=True):
        """
        Load model parameters, optimizer state, training progress and optionally replay buffer
        Returns True if loading was successful, False otherwise
        """
        print('load_checkpoint')
        checkpoint_path = os.path.join(checkpoint_dir, f"{filename_prefix}_checkpoint.pth")
        buffer_path = os.path.join(checkpoint_dir, f"{filename_prefix}_replay_buffer.pkl")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            print('no')
            return False
        
        try:
            # Load model checkpoint - explicitly set device
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            
            # Ensure models are on the correct device
            self.q_network = self.q_network.to(self.device)
            self.target_network = self.target_network.to(self.device)
            
            # Load optimizer state and move to correct device
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Fix optimizer's device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            # Load training progress variables
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
    
            # Optionally load other hyperparameters if they exist
            if 'gamma' in checkpoint:
                self.gamma = checkpoint['gamma']
            if 'lr' in checkpoint:
                self.lr = checkpoint['lr']
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Current epsilon: {self.epsilon}, Steps: {self.step_count}")
            
            # Load replay buffer if requested
            if load_replay_buffer and os.path.exists(buffer_path):
                try:
                    with open(buffer_path, 'rb') as f:
                        buffer_data = pickle.load(f)
                        # Replace the current replay buffer data
                        self.replay_buffer.data.clear()
                        self.replay_buffer.data.extend(buffer_data)
                    print(f"Loaded replay buffer with {len(self.replay_buffer.data)} experiences")
                except Exception as e:
                    print(f"Failed to load replay buffer: {e}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False
        
    def get_action(self, state, deterministic=True):
        # TODO: Implement the action selection
        img = torch.tensor(np.stack(state), dtype=torch.float32,device = self.device)  # Use float32 for image data
        img = img.unsqueeze(0)
        _epsilon = self.epsilon
        if deterministic or np.random.rand() > _epsilon:
            self.q_network.eval()
            with torch.no_grad():
                result = self.q_network(img)
            return torch.argmax(result).item()
        else:
            return np.random.randint(self.action_size)

    def update(self):
        # TODO: Implement hard update or soft update
        # implement hard update
        if self.step_count%self.target_update_frequency:
            return
        self.target_network.load_state_dict(self.q_network.state_dict())
        return

    def train(self):
        
        self.step_count += 1
        if self.step_count < self.epsilon_decay_steps:
            self.epsilon = 1 - (1-self.epsilon_min)*self.step_count / self.epsilon_decay_steps
        else:
            self.epsilon = self.epsilon_min
        if self.step_count % self.update_frequency:
            return
        self.q_network.train()
        self.target_network.eval()
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        # Stack into arrays
        #imgs = np.stack(imgs)  # Shape: (batch_size, 4, 84, 84)
        #lives = np.stack(lives)  # Shape: (batch_size, 4)
        
        # Convert to PyTorch tensors
        #imgs = torch.tensor(imgs, dtype=torch.float32, device=self.device)  # Shape: (batch_size, 4, 84, 84)
        #lives = torch.tensor(lives, dtype=torch.float32, device=self.device)  # Shape: (batch_size, 4)
        #states = [imgs,lives]
        states = torch.tensor(np.stack(states),dtype=torch.float32,device=self.device)
        next_states = torch.tensor(np.stack(next_states),dtype=torch.float32,device=self.device)
        # Process next_states similarly
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            best_actions = torch.argmax(self.q_network(next_states), dim=-1)
            target = self.gamma * torch.gather(self.target_network(next_states), dim=-1, index=best_actions.unsqueeze(-1)).squeeze(-1) * (1 - dones) + rewards
        # compute loss
        output = torch.gather(self.q_network(states), dim=-1, index=actions_tensor).squeeze(-1)
        
        loss = self.loss_fn(output, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update()
        # TODO: Update target network periodically
        
    def validate(self, state, action, next_state, reward, done):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_val = self.q_network(state_tensor)[0, action]
            next_max_q = self.target_network(next_state_tensor).max(dim=1)[0].item()
            target = reward + self.gamma * next_max_q * (1 - done)
            target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)
            loss = self.loss_fn(q_val, target_tensor)
        return loss
