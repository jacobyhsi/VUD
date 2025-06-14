import numpy as np

class UCB1_Algorithm:
    def __init__(self, num_arms: int, c: float = 2.0):
        """
        Initialize the UCB algorithm with a number of arms and a confidence parameter c.
        """
        self.num_arms = num_arms
        self.c = c
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_counts = 0
        self.total_reward = 0.0

    def select_arm(self):
        """
        Select the arm to pull based on the UCB algorithm.
        """
        if self.total_counts < self.num_arms:
            return self.total_counts
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """
        Update the algorithm with the result of pulling an arm.
        """
        self.counts[arm] += 1
        self.total_counts += 1
        self.total_reward += reward
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        
    def get_average_reward(self):
        """
        Get the average reward of the algorithm.
        """
        return self.total_reward / self.total_counts if self.total_counts > 0 else 0.0
    
    def get_uncertainty(self, arm: int):
        """
        Get the uncertainty of the algorithm.
        """
        if self.counts[arm] == 0:
            return float(1e5)
        return self.c * np.sqrt(np.log(self.total_counts) / (self.counts[arm] + 1e-5))