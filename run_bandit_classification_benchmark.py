import os
import re
import argparse
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.bandit import get_bandit, ButtonsBandit
from src.bandit_algorithms import UCB1_Algorithm
from src.chat import chat_response_only

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Running Bandit Classification Benchmark')

parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)
parser.add_argument("--max_tokens", default=1000, type=int)
parser.add_argument("--temperature", default=0.0, type=float)

parser.add_argument("--bandit_name", default="buttons", type=str)
parser.add_argument("--bandit_num_arms", default=5, type=int)
parser.add_argument("--bandit_midpoint", default=0.5, type=float)
parser.add_argument("--bandit_gap", default=0.2, type=float)
parser.add_argument("--bandit_seed", default=0, type=int)

parser.add_argument("--num_trials", default=200, type=int)

parser.add_argument("--run_name", default="test")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--verbose_output", default=0, type=int)

args = parser.parse_args()

@dataclass
class BanditClassificationBenchmarkExperimentConfig:
    model_name: str
    model_port: str
    model_ip: str
    max_tokens: int
    temperature: float

    bandit_name: str
    bandit_num_arms: int
    bandit_midpoint: float
    bandit_gap: float
    bandit_seed: int

    num_trials: int

    run_name: str
    save_directory: str
    num_api_calls_save_value: int

    verbose_output: int


SYSTEM_PROMPT = """You are a bandit algorithm in a room with 5 buttons labeled blue, green, red, yellow, purple. Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the buttons could be different. For each button, when you press it, you will get a reward that is sampled from the button’s associated distribution. You have {time} time steps and, on each time step, you can choose any button and receive the reward. Your goal is to maximize the total reward over the 10 time steps.

At each time step, I will show you a summary of your past choices and rewards. Then you must make the next choice, which must be exactly one of blue, green, red, yellow, purple. Let’s think step by step to make sure we make a good choice. You must provide your final answer within the tags <Answer>COLOR</Answer> where COLOR is one of blue, green, red, yellow, purple."""

USER_PROMPT = """So far you have played {trial} times with your past choices and rewards summarized as follows:
blue button: pressed {arm_count_0} times{average_reward_0}
green button: pressed {arm_count_1} times{average_reward_1}
red button: pressed {arm_count_2} times{average_reward_2}
yellow button: pressed {arm_count_3} times{average_reward_3}
purple button: pressed {arm_count_4} times{average_reward_4}

Which button will you choose next? Remember, YOU MUST provide your final answer within the tags <Answer>COLOR</Answer> where COLOR is one of blue, green, red, yellow, purple. Let’s think step by step to make sure we make a good choice."""

COLOUR_ACTIONS_TO_ARM = {
    "blue": 0,
    "green": 1,
    "red": 2,
    "yellow": 3,
    "purple": 4
}

class BanditClassificationBenchmarkExperiment:
    def __init__(self, config: BanditClassificationBenchmarkExperimentConfig):
        self.config = config
    
        self.create_bandit()
                
        self.num_api_calls = self.config.num_api_calls_save_value
        
    def create_bandit(self):
        self.bandit: ButtonsBandit = get_bandit(
            bandit_name=self.config.bandit_name,
            num_arms=self.config.bandit_num_arms,
            gap=self.config.bandit_gap,
            midpoint=self.config.bandit_midpoint,
            seed=self.config.bandit_seed,
        )
        
        self.ucb1_algorithm = UCB1_Algorithm(
            num_arms=self.config.bandit_num_arms,
            c=0
        )
        
        self.label_keys = self.bandit.get_reward_space()
        self.action_space = self.bandit.get_action_space()
        
        if isinstance(self.bandit, ButtonsBandit):
            print(f"Best arm: {self.bandit.best_arm}")

        self.feature_columns = []
            
        self.num_trials = self.config.num_trials
        
        self.D_rows: pd.DataFrame = None

    def save_D_rows(self):
        if not os.path.exists(f"results/bandits/{self.config.bandit_name}/benchmark/{self.config.save_directory}"):
            os.makedirs(f"results/bandits/{self.config.bandit_name}/benchmark/{self.config.save_directory}")
        self.D_rows.to_csv(f"results/bandits/{self.config.bandit_name}/benchmark/{self.config.save_directory}/D_{self.config.run_name}.csv", index=False)
        
    @staticmethod
    def extract_colour(text):
        match = re.search(r'<Answer>(.*?)</Answer>', text, re.DOTALL | re.IGNORECASE)
        if match:
            output_str = match.group(1).strip()
            return output_str.lower()
        else:
            print(text)
            print("Could not find output tags in the response.")
            raise ValueError("Invalid response format.")
    
    def get_single_trial_action(self, trial: int):
        arm_counts = self.ucb1_algorithm.counts
        average_rewards = self.ucb1_algorithm.values
        
        system_prompt = SYSTEM_PROMPT.format(time=self.config.num_trials)
        user_prompt = USER_PROMPT.format(
            trial=trial,
            arm_count_0=arm_counts[0],
            arm_count_1=arm_counts[1],
            arm_count_2=arm_counts[2],
            arm_count_3=arm_counts[3],
            arm_count_4=arm_counts[4],
            average_reward_0=f" with average reward {average_rewards[0]:.2f}" if arm_counts[0] > 0 else "",
            average_reward_1=f" with average reward {average_rewards[1]:.2f}" if arm_counts[1] > 0 else "",
            average_reward_2=f" with average reward {average_rewards[2]:.2f}" if arm_counts[2] > 0 else "",
            average_reward_3=f" with average reward {average_rewards[3]:.2f}" if arm_counts[3] > 0 else "",
            average_reward_4=f" with average reward {average_rewards[4]:.2f}" if arm_counts[4] > 0 else "",
        )
        
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        
        if self.config.verbose_output == 1:
            print(f"\nTrial {trial + 1}/{self.config.num_trials}")
            print(f"System Prompt: {system_prompt}")
            print(f"User Prompt: {user_prompt}")
            
        
        for attempt in range(5):
            try: 
                seed = self.num_api_calls
                
                text_output = chat_response_only(
                    message=full_prompt,
                    seed=seed,
                    max_tokens=self.config.max_tokens + attempt * 500,
                    temperature=self.config.temperature,
                    model=self.config.model_name,
                    port=self.config.model_port,
                    ip=self.config.model_ip
                )
                
                self.num_api_calls += 1
                
                if self.config.verbose_output == 1:
                    print(f"Chat Output: {text_output}")
                
                colour_action = self.extract_colour(text_output)
                
                print(f"Colour Action: {colour_action}")
                
                arm = COLOUR_ACTIONS_TO_ARM[colour_action]
                
                return arm, (attempt+1)
            except:
                print(f"Error in API call, retrying... (attempt {attempt + 1})")
                
        raise Exception("Failed to get a valid action after 5 attempts.")
                    
    def single_trial(self, trial: int):
        print(f"\nTrial {trial + 1}/{self.config.num_trials}")
        
        action_taken, attempts = self.get_single_trial_action(trial)
        
        reward = self.bandit.get_reward(action_taken)
        
        regret = self.bandit.get_optimal_mean_reward() - reward

        print(f"Action: {action_taken}; Reward: {reward}; Regret: {regret}")
        
        self.ucb1_algorithm.update(action_taken, reward)
        
        trial_df = pd.DataFrame(
            {
                "trial": [trial],
                "action": [action_taken],
                "reward": [reward],
                "regret": [regret],
                "optimal_action": [self.bandit.optimal_action()],
                "attempts": [attempts],
            }
        )
        
        if trial == 0:
            self.D_rows = trial_df
        else:
            self.D_rows = pd.concat([self.D_rows, trial_df], ignore_index=True)
            
        self.save_D_rows()
        
    def run_experiment(self):
        for trial in range(self.config.num_trials):
            self.single_trial(trial)
            
        print(f"\nTotal API calls: {self.num_api_calls}")
        
def main():
    config = BanditClassificationBenchmarkExperimentConfig(**vars(args))
    
    experiment = BanditClassificationBenchmarkExperiment(config)
    
    experiment.run_experiment()

if __name__ == "__main__":
    main()