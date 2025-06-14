import numpy as np
import pandas as pd

class MultiArmBandit:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
    
    def get_reward(self, action: int|str, **kwargs) -> float|int:
        """
        Get the reward for a given action in a given context.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_optimal_mean_reward(self, **kwargs) -> float|int:
        """
        Get the optimal mean reward for a given action in a given context.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_context_feature_cols(self) -> dict:
        """
        Get the context features for the bandit.
        """
        return []
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_next_context(self) -> pd.DataFrame:
        """
        Get the context for the bandit.
        """
        return None
    
    def optimal_action(self, **kwargs) -> int:
        """
        Get the best action for the bandit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class ClassificationBandit(MultiArmBandit):
    def get_reward_space(self) -> list:
        """
        Get the reward values for the bandit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class RegressionBandit(MultiArmBandit):
    pass   

class ButtonsBandit(ClassificationBandit):
    def __init__(self, num_arms: int = 2, midpoint: float = 0.5, gap: float = 0.2, seed: int = 0, **kwargs):
        """
        Initialize the bandit with a number of buttons.
        """
        super().__init__(seed)
        self.num_arms = num_arms
        self.best_arm = self.rng.integers(0, num_arms)
        self.midpoint = midpoint
        self.gap = gap
    
    def get_reward(self, action: int|str) -> float|int:
        """
        Get the reward for a given button action.
        """
        
        if isinstance(action, str):
            action = int(action)
        
        if action not in range(self.num_arms):
            raise ValueError(f"Action {action} is not a valid arm.")
        
        if action == self.best_arm:
            reward = self.rng.binomial(1, self.midpoint + self.gap*0.5)
        else:
            reward = self.rng.binomial(1, self.midpoint - self.gap*0.5)

        return reward
    
    def get_optimal_mean_reward(self) -> float|int:
        """
        Get the optimal mean reward for a given button action.
        """
        
        return self.midpoint + self.gap*0.5
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        
        return list(range(self.num_arms))
    
    def get_reward_space(self):
        return ["0", "1"]
    
    def optimal_action(self, **kwargs) -> int:
        return self.best_arm