import json
import os
import hashlib
from typing import Dict, List, Any
import torch
import math
import json
from verl.protocol import DataProto


# Adaptive batch size calculation
def get_next_target_size(b_delta: int, default_batch_size: int, zero_variance_ratio: float, beta: float = 1.25) -> int:
    
    expected_samples = (beta * b_delta) / max(1.0 - zero_variance_ratio, 0.01)
    return max(min(default_batch_size, math.ceil(expected_samples)), 64) # The max here is in the original implementation but undocumented in the paper


# Class to keep track of an accumulating batch of DataProto objects along with streak metadata
class DataProtoBuffer:
    def __init__(self):
        self.buffer = []
        self.total_size = 0

    def append(self, data):
        self.buffer.append(data)
        self.total_size += len(data)

    def __len__(self):
        return self.total_size

    
    # Extract target_size elements from the buffer
    def pop_batch(self, target_size: int):
        assert len(self) >= target_size, "Buffer does not have enough elements"
        
        needed_elements = []
        current_collected = 0
        
        # Collect the batch
        while current_collected < target_size:
            front = self.buffer.pop(0)
            front_len = len(front)
            
            if current_collected + front_len <= target_size:
                needed_elements.append(front)
                current_collected += front_len
            else:
                remaining_needed = target_size - current_collected
                
                front_slice = front[0:remaining_needed]
                front_keep = front[remaining_needed:front_len]
                
                needed_elements.append(front_slice)
                self.buffer.insert(0, front_keep)
                current_collected += remaining_needed
        
        self.total_size -= target_size
        return DataProto.concat(needed_elements)

class PromptHistoryTracker:
    def __init__(self, target_easy_ratio=0.083, target_hard_ratio=0.167, delta_p=0.01):
        self.history = {}
        self.target_easy_ratio = target_easy_ratio
        self.target_hard_ratio = target_hard_ratio
        self.delta_p = delta_p
        
        self.p_e_easy = 0.0
        self.p_e_hard = 0.0

    def get_prompt_id(self, prompt_text) -> str:
        if isinstance(prompt_text, str):
            return hashlib.md5(prompt_text.encode('utf-8')).hexdigest()
        else:
            return hashlib.md5(json.dumps(prompt_text, sort_keys=True).encode('utf-8')).hexdigest()

    def pre_rollout_probabilistic_filter(self, data, prompts_decoded: List[str]):
        import torch
        surviving_indices = []
        for idx, pt in enumerate(prompts_decoded):
            pid = self.get_prompt_id(pt)
            if pid in self.history:
                streak_type = self.history[pid]["streak_type"]
                streak_length = self.history[pid]["streak_length"]
                if streak_type == "easy":
                    skip_prob = 1.0 - max(min(self.p_e_easy ** streak_length, 1.0), 0.01)
                elif streak_type == "hard":
                    skip_prob = 1.0 - max(min(self.p_e_hard ** streak_length, 1.0), 0.01)
                else:
                    skip_prob = 0.0
                
                if torch.rand(1).item() > skip_prob:
                    surviving_indices.append(idx)
            else:
                # new prompt always survive
                surviving_indices.append(idx)
        
        if len(surviving_indices) == 0:
            return None
            
        return data[surviving_indices]

    # Filter out zero variance rewards 
    def post_rollout_filter(self, data, rewards: torch.Tensor, n: int):
        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1)
            
        num_prompts = len(rewards) // n
        rewards_2d = rewards.view(num_prompts, n)
        
        min_r = rewards_2d.min(dim=-1)[0]
        max_r = rewards_2d.max(dim=-1)[0]
        
        variance_mask = max_r > min_r
        zero_variance_ratio = 1.0 - (variance_mask.sum().item() / max(num_prompts, 1))
        
        flat_mask = variance_mask.repeat_interleave(n)
        
        if flat_mask.any():
            surviving_indices = flat_mask.nonzero(as_tuple=True)[0].tolist()
            return data[surviving_indices], zero_variance_ratio
        else:
            return None, zero_variance_ratio

    # Update the history and streaks     
    def update(self, epoch: int, prompt_texts: List[str], rewards: List[List[float]]):
        for pt, r_list in zip(prompt_texts, rewards):
            pid = self.get_prompt_id(pt)
            if pid not in self.history:
                self.history[pid] = {
                    "rewards": {},
                    "streak_type": "none",
                    "streak_length": 0
                }
            
            # Save rewards
            self.history[pid]["rewards"][str(epoch)] = r_list
            
            # Check variance
            if len(r_list) > 1:
                variance = sum((r - sum(r_list)/len(r_list))**2 for r in r_list) / (len(r_list) - 1)
            else:
                variance = 0.0
            
            mean_reward = sum(r_list) / len(r_list) if r_list else 0.0
            
            # Classify as easy or hard
            if variance == 0.0:
                is_easy = mean_reward > 0.0
                current_type = "easy" if is_easy else "hard"
                
                if self.history[pid]["streak_type"] == current_type:
                    self.history[pid]["streak_length"] += 1
                else:
                    self.history[pid]["streak_type"] = current_type
                    self.history[pid]["streak_length"] = 1
            else:
                self.history[pid]["streak_type"] = "none"
                self.history[pid]["streak_length"] = 0

    def adjust_p_e(self, observed_easy_ratio: float, observed_hard_ratio: float):
        if observed_easy_ratio >= self.target_easy_ratio:
            self.p_e_easy = max(0.0, self.p_e_easy - self.delta_p)
        else:
            self.p_e_easy = min(1.0, self.p_e_easy + self.delta_p)
            
        if observed_hard_ratio >= self.target_hard_ratio:
            self.p_e_hard = max(0.0, self.p_e_hard - self.delta_p)
        else:
            self.p_e_hard = min(1.0, self.p_e_hard + self.delta_p)

    def state_dict(self):
        return {
            "history": self.history,
            "p_e_easy": self.p_e_easy,
            "p_e_hard": self.p_e_hard
        }

    def load_state_dict(self, state_dict):
        self.history = state_dict.get("history", {})
        self.p_e_easy = state_dict.get("p_e_easy", 0.0)
        self.p_e_hard = state_dict.get("p_e_hard", 0.0)

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.state_dict(), f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.load_state_dict(json.load(f))
