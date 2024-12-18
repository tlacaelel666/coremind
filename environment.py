import gym
import torch
import torch.nn as nn

# Constants for readability and reuse
DEFAULT_PRIORITY = 1.0
HIDDEN_DIM = 128
DISCOUNT_FACTOR = 0.99
STATE_DIM = 4
ACTION_DIM = 2
NUM_SAMPLES = 1
LEARNING_RATE = 0.01
NUM_EPISODES = 1000
TENSOR_DTYPE = torch.float32  # Introduced for consistent tensor data type
INFO_KEY_DEFAULT = 0.0  # Default value for 'info' tensor if not used


class ConditionalActionsSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditions = {}
        self.priorities = {}
        self.actions = {}

    def add_action(self, *, condition, action, priority=DEFAULT_PRIORITY):
        """Adds a conditional action with a priority."""
        self.conditions[condition] = action
        self.priorities[condition] = priority


class Action:
    def __init__(self, environment, conditional_actions_system, policy):
        self.environment = environment
        self.conditional_actions_system = conditional_actions_system
        self.policy = policy
        self.reward_log = []  # Renamed from reward_log for clarity

    def select_action_using_policy(self, action_policy, state):
        return self._select_action_from_policy(action_policy, state)

    def execute_environment_action(self, action):
        new_state, reward, done, _ = self.environment.step_and_convert(action)  # Renamed
        self._record_reward(reward)
        return new_state, reward, done

    def perform_policy_action(self, state):
        action = self.select_action_using_policy(self.policy, state)
        return self.execute_environment_action(action)

    def _record_reward(self, reward):
        self.reward_log.append(reward)
        self.conditional_actions_system.store_experience(reward, self.reward_log)

    @staticmethod
    def _select_action_from_policy(action_policy, state):
        with torch.no_grad():
            probabilities, _ = action_policy(state)
        return torch.multinomial(probabilities, NUM_SAMPLES).item()


class CartPoleEnvironment:
    """A wrapper for the CartPole Gym environment."""
    ENV_NAME = "CartPole-v1"

    @staticmethod
    def _convert_to_tensor(data):
        return torch.tensor(data, dtype=TENSOR_DTYPE)

    @staticmethod
    def _convert_env_step_result(state, reward, done, info):
        """Convert each element from the environment's result to tensors."""
        state = CartPoleEnvironment._convert_to_tensor(state)
        reward = CartPoleEnvironment._convert_to_tensor(reward)
        done = CartPoleEnvironment._convert_to_tensor(done)
        info = CartPoleEnvironment._convert_to_tensor(info or INFO_KEY_DEFAULT)
        return state, reward, done, info

    def __init__(self, env_name=ENV_NAME):
        self.env = gym.make(env_name)
        self.reset()

    def reset(self):
        state, _info = self.env.reset()
        return self._convert_to_tensor(state)

    def step(self, action):
        return self.env.step(action)

    def step_and_convert(self, action):
        """Perform a step and convert the result to tensors."""
        state, reward, done, info = self.env.step(action)
        return self._convert_env_step_result(state, reward, done, info)


class ActorCriticA2C(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU())
