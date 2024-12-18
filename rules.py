import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class A2CAgent:
    # Constants
    LEARNING_RATE = 0.001
    ACTIVATION_FUNCTIONS = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
    }

    def __init__(self, environment, state_dim, action_dim, hidden_dim, gamma=0.99, activation_fn="relu",
                 output_dim=None):
        self.environment = environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.output_dim = output_dim if output_dim else state_dim
        self.activation_fn = self._get_activation_function(activation_fn)

        # Models & Optimizers
        self.actor_critic = self._build_actor_critic()
        self.optimizer_actor_critic = optim.Adam(self.actor_critic.parameters(), lr=self.LEARNING_RATE)
        self.prediction_model = self._build_prediction_model(self.state_dim, self.hidden_dim, self.output_dim)
        self.optimizer_prediction = optim.Adam(self.prediction_model.parameters(), lr=self.LEARNING_RATE)

        # Experiences & Metrics
        self.experiences = {"log_probs": [], "values": [], "rewards": []}
        self.metrics = {"trajectory": [], "entropy": [], "coherence": [], "quality": []}

        # Graphs
        self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 12))
        self.canvas = None

    def _get_activation_function(self, name):
        """Helper to retrieve activation function."""
        return self.ACTIVATION_FUNCTIONS.get(name.lower(), nn.ReLU())  # Default to ReLU

    def _build_actor_critic(self):
        """Builds the Actor-Critic model."""
        return self._build_interface_model(
            self.state_dim, self.hidden_dim, self.action_dim, is_actor_critic=True
        )

    def _build_prediction_model(self, input_dim, hidden_dim, output_dim):
        """Builds the prediction model."""
        return self._build_interface_model(
            input_dim, hidden_dim, output_dim, is_actor_critic=False
        )

    def _build_interface_model(self, input_dim, hidden_dim, output_dim, is_actor_critic):
        """Helper to build both Actor-Critic and Prediction models."""

        # noinspection PyPep8Naming
        class InterfaceModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, policy=None, value=None, out=None):
                super(InterfaceModel, self).__init__()
                self.fc = nn.Linear(input_dim, hidden_dim)
                self.activation = activation_fn
                self.policy = policy
                self.value = value
                self.out = out



                def _build_interface_model(self, input_dim, hidden_dim, output_dim, is_actor_critic):
                    """Helper to build both Actor-Critic and Prediction models."""
                    # Configure the layers based on model type
                    policy_layer = nn.Linear(hidden_dim, output_dim) if is_actor_critic else None
                    value_layer = nn.Linear(hidden_dim, 1) if is_actor_critic else None
                    output_layer = nn.Linear(hidden_dim, output_dim) if not is_actor_critic else None
                
                    # The external InterfaceModel class is utilized here.
                    return InterfaceModel(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        activation_fn=self._get_activation_function(self.activation_fn),
                        policy=policy_layer,
                        value=value_layer,
                        out=output_layer
                    )


class InterfaceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, policy=None, value=None, out=None):
        super(InterfaceModel, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.activation = activation_fn
        self.policy = policy
        self.value = value
        self.out = out

    def forward(self, state):
        x = self.activation(self.fc(state))
        if self.policy and self.value:  # Actor-Critic
            policy_logits = torch.softmax(self.policy(x), dim=-1)
            value = self.value(x)
            return policy_logits, value
        elif self.out:  # Prediction Model
            return self.out(x)
        else:
            raise ValueError("Invalid model configuration.")


        return InterfaceModel(input_dim, hidden_dim, output_dim, self.activation_fn, is_actor_critic)

        R = 0
        discounted = []
        for reward in reversed(rewards):
         R = reward + self.gamma * R
        discounted.insert(0, R)
        return torch.tensor(discounted, dtype=torch.float32)

    def store_experience(self, state, action, reward):
        """Stores experiences."""
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action_probs, state_value = self.actor_critic(state_tensor)
        log_prob = torch.log(action_probs.squeeze(0)[action])
        self.experiences["log_probs"].append(log_prob)
        self.experiences["values"].append(state_value)
        self.experiences["rewards"].append(reward)

    def update_graphs(self):
        """Updates graphs based on the current metrics."""
        metrics = ["trajectory", "entropy", "coherence", "quality"]
        colors = ["blue", "orange", "green", "purple"]
        titles = ["Trajectory", "Entropy", "Coherence", "Quality"]
        for i, metric in enumerate(metrics):
            self.axs[i].cla()
            self.axs[i].plot(self.metrics[metric], color=colors[i], label=titles[i])
            self.axs[i].legend()
        self.fig.canvas.draw()
