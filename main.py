import random
import tkinter as tk
from tkinter import ttk, scrolledtext
import torch
import torch.nn as nn
import torch.optim as optim
"""
     Archivo principal de ejecución del proyecto.
     Permite iniciar la aplicación, gestionar sus módulos principales y centralizar la ejecución.
"""
# Actor-Critic Model for A2C
class ActorCriticA2C(nn.Module):
    def __init__(self, input_size, action_size, hidden_layer_size):
        super(ActorCriticA2C, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.policy_layer = nn.Linear(hidden_layer_size, action_size)
        self.value_layer = nn.Linear(hidden_layer_size, 1)

    def forward(self, state):
        hidden = torch.relu(self.input_layer(state))
        policy_logits = self.policy_layer(hidden)
        value = self.value_layer(hidden)
        return torch.softmax(policy_logits, dim=-1), value


# Prediction Module (RNN-Based)
class PredictionModule(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(PredictionModule, self).__init__()
        self.rnn_layer = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        _, (hidden_state, _) = self.rnn_layer(x)
        return self.output_layer(hidden_state[-1])


# Main A2C Application
class A2CApplication:
    def __init__(self, title="A2C Agent in Binary Object Environment"):
        self.title = title
        self.state_dim = 1
        self.action_dim = 4
        self.hidden_dim = 128
        self.experiences = {"log_probs": [], "values": [], "rewards": []}

        # Initialize UI and Models
        self._initialize_ui()
        self.actor_critic = ActorCriticA2C(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        self.prediction_module = PredictionModule(self.state_dim, self.hidden_dim, 1)
        self.prediction_optimizer = optim.Adam(self.prediction_module.parameters(), lr=0.001)

    def train_agent(self):
        """Train the A2C agent."""
        for epoch in range(500):
            # Clear experiences every training epoch
            self.experiences = {"log_probs": [], "values": [], "rewards": []}

            # Simulate environment and experience collection
            for _ in range(10):  # Assume 10 steps per episode
                state = random.randint(0, 3)  # Example state
                action = self.select_action(state)
                reward = random.uniform(-1, 1)  # Example reward
                self.store_experience(state, action, reward)

            # Perform optimization using collected experiences
            discounted_rewards = self.calculate_discounted_rewards(self.experiences["rewards"], 0.99)
            log_probs = torch.stack(self.experiences["log_probs"])  # Stack experiences
            values = torch.cat(self.experiences["values"])  # Concatenate value tensors

            # Calculate actor and critic losses
            actor_loss, critic_loss = self._calculate_losses(log_probs, values, discounted_rewards)

            # Reset gradients and perform backpropagation
            self.optimizer.zero_grad()  # Clear gradients
            (actor_loss + critic_loss).backward()  # Backward pass
            self.optimizer.step()  # Update parameters

            # Update feedback in the UI periodically
            if (epoch + 1) % 100 == 0:
                self.feedback_text.insert(tk.END, f"Epoch {epoch + 1}: Loss: {actor_loss + critic_loss:.4f}\n")
                self.feedback_text.see(tk.END)

        self.root.mainloop()

    def _initialize_ui(self):
        """Initializes and sets up the GUI."""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.feedback_text = self._setup_feedback_area()
        ttk.Button(self.root, text="Train Agent", command=self.train_agent).grid(column=0, row=1, padx=10, pady=5)

    def _setup_feedback_area(self):
        """Creates a feedback area for UI."""
        feedback_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=60, height=20)
        feedback_text.grid(column=0, row=0, padx=10, pady=10)
        return feedback_text

    def store_experience(self, state, action, reward):
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action_probs, state_value = self.actor_critic(state_tensor)
        log_prob = torch.log(action_probs[action])
        self.experiences["log_probs"].append(log_prob)
        self.experiences["values"].append(state_value)
        self.experiences["rewards"].append(reward)

    def select_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
        return action_probs.multinomial(1).item()

    @staticmethod
    def calculate_discounted_rewards(rewards, gamma):
        """Calculate discounted rewards."""
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return torch.tensor(discounted_rewards, dtype=torch.float32)

    @staticmethod
    def _calculate_losses(log_probs, values, discounted_rewards):
        """Extracted loss calculation logic."""
        advantages = discounted_rewards - values.squeeze()
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        return actor_loss, critic_loss

if __name__ == "__main__":
    app = A2CApplication()
    app.train_agent()
    app.root.mainloop()