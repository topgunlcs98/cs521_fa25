import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import argparse

OUTPUT_DIR = "./bonus_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class RewardModel:
    def __init__(self, name: str):
        self.name = name

    def compute_reward(self, text: str) -> float:
        raise NotImplementedError


class GoodRewardModel(RewardModel):
    def __init__(self):
        super().__init__("Good Reward (No Hacking)")

    def compute_reward(self, text: str) -> float:
        reward = 0.0

        length = len(text)
        if 50 <= length <= 200:
            reward += 1.0
        elif 200 < length <= 300:
            reward += 0.5
        elif length > 300:
            reward -= 0.5

        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio  # Higher reward for diverse vocabulary

        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
        reward += alpha_ratio * 0.5

        if text.strip().endswith(('.', '!', '?')):
            reward += 0.3

        return reward


class HackableRewardModel(RewardModel):

    def __init__(self):
        super().__init__("Hackable Reward (With Hacking)")
        self.target_keywords = ['good', 'great', 'excellent', 'amazing', 'wonderful']

    def compute_reward(self, text: str) -> float:
        reward = 0.0
        reward += min(len(text) / 100, 5.0)

        text_lower = text.lower()
        for keyword in self.target_keywords:
            count = text_lower.count(keyword)
            reward += count * 0.5
        reward += text.count('!') * 0.2

        return reward


def prepare_dataset(num_samples: int = 100, max_dataset_size: int = 2000):
    dataset = load_dataset("trl-lib/Capybara", split=f"train[:{max_dataset_size}]")
    prompts = []
    for i in range(min(num_samples, len(dataset))):
        idx = i % len(dataset)
        messages = dataset[idx]['messages']
        user_message = None
        for msg in messages:
            if msg['role'] == 'user':
                user_message = msg['content']
                break
        if user_message:
            prompt = user_message[:300] if len(user_message) > 300 else user_message
            prompts.append(prompt)
    print(f"Prepared {len(prompts)} prompts for training")
    return prompts


def train_with_ppo(model_name: str, reward_model: RewardModel,
                   prompts: List[str], num_epochs: int = 3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        seed=42,
    )
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    device = ppo_trainer.accelerator.device

    all_rewards = []
    all_generated_texts = []
    batch_size = config.batch_size
    for epoch in range(num_epochs):
        epoch_rewards = []
        for batch_idx in range(0, len(prompts[:20]), batch_size):
            batch_prompts = prompts[batch_idx:min(batch_idx + batch_size, len(prompts[:20]))]
            batch_query_tensors = []
            batch_response_tensors = []
            batch_rewards = []
            for i, prompt in enumerate(batch_prompts):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated_text = tokenizer.decode(output[0][input_ids.shape[1]:],
                                                 skip_special_tokens=True)
                reward = reward_model.compute_reward(generated_text)
                reward_tensor = torch.tensor([reward], device=device)
                epoch_rewards.append(reward)
                sample_idx = batch_idx + i
                if sample_idx < 5:
                    all_generated_texts.append({
                        'epoch': epoch,
                        'prompt': prompt[:100] + "...",
                        'generated': generated_text,
                        'reward': reward
                    })
                batch_query_tensors.append(input_ids.squeeze())
                batch_response_tensors.append(output.squeeze()[input_ids.shape[1]:])
                batch_rewards.append(reward_tensor)
            if len(batch_query_tensors) > 0:
                stats = ppo_trainer.step(batch_query_tensors, batch_response_tensors, batch_rewards)

                avg_batch_reward = np.mean([r.item() for r in batch_rewards])
                print(f"  Batch {batch_idx//batch_size + 1}: Avg Reward = {avg_batch_reward:.3f}")

        avg_reward = np.mean(epoch_rewards)
        all_rewards.append(avg_reward)
        print(f"  Epoch {epoch + 1} Average Reward: {avg_reward:.3f}")

    return {
        'model': model,
        'rewards': all_rewards,
        'examples': all_generated_texts
    }


def analyze_reward_hacking(good_results: Dict, hackable_results: Dict):
    print(f"\n{'='*70}")
    print("Reward Hacking Comparison")
    print(f"{'='*70}\n")

    # Compare final rewards
    print("1. REWARD PROGRESSION:")
    print(f"Good Reward Model - Final Average: {good_results['rewards'][-1]:.3f}")
    print(f"Hackable Reward Model - Final Average: {hackable_results['rewards'][-1]:.3f}")

    # Plot reward curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    epochs = range(1, len(good_results['rewards']) + 1)
    plt.plot(epochs, good_results['rewards'], 'b-o', label='Good Reward Model', linewidth=2)
    plt.plot(epochs, hackable_results['rewards'], 'r-s', label='Hackable Reward Model', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Reward Progression During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Analyze example generations
    print("\n2. EXAMPLE GENERATIONS (Final Epoch):")
    print("\n   Good Reward Model Examples:")
    for i, ex in enumerate(good_results['examples'][-3:]):
        if ex['epoch'] == len(good_results['rewards']) - 1:
            print(f"   [{i+1}] Generated: {ex['generated'][:100]}...")
            print(f"       Reward: {ex['reward']:.3f}\n")

    print("\n   Hackable Reward Model Examples:")
    for i, ex in enumerate(hackable_results['examples'][-3:]):
        if ex['epoch'] == len(hackable_results['rewards']) - 1:
            print(f"   [{i+1}] Generated: {ex['generated'][:100]}...")
            print(f"       Reward: {ex['reward']:.3f}\n")

    # Analyze text characteristics
    print("\n3. TEXT CHARACTERISTICS ANALYSIS:")

    good_final_texts = [ex['generated'] for ex in good_results['examples']
                        if ex['epoch'] == len(good_results['rewards']) - 1]
    hack_final_texts = [ex['generated'] for ex in hackable_results['examples']
                        if ex['epoch'] == len(hackable_results['rewards']) - 1]

    if good_final_texts and hack_final_texts:
        good_avg_len = np.mean([len(t) for t in good_final_texts])
        hack_avg_len = np.mean([len(t) for t in hack_final_texts])

        good_repetition = []
        hack_repetition = []

        for text in good_final_texts:
            words = text.split()
            if words:
                good_repetition.append(len(set(words)) / len(words))

        for text in hack_final_texts:
            words = text.split()
            if words:
                hack_repetition.append(len(set(words)) / len(words))

        print(f"   Good Model - Avg Length: {good_avg_len:.1f} chars")
        print(f"   Hackable Model - Avg Length: {hack_avg_len:.1f} chars")
        print(f"   Good Model - Unique Word Ratio: {np.mean(good_repetition):.3f}")
        print(f"   Hackable Model - Unique Word Ratio: {np.mean(hack_repetition):.3f}")

        # Plot text statistics
        plt.subplot(1, 2, 2)
        categories = ['Avg Length', 'Unique Word\nRatio']
        good_stats = [good_avg_len / 10, np.mean(good_repetition)]  # Normalize length
        hack_stats = [hack_avg_len / 10, np.mean(hack_repetition)]

        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width/2, good_stats, width, label='Good Reward', color='blue', alpha=0.7)
        plt.bar(x + width/2, hack_stats, width, label='Hackable Reward', color='red', alpha=0.7)
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Text Characteristics Comparison')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reward_hacking_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n   Saved plot to: {OUTPUT_DIR}/reward_hacking_analysis.png")



def main(model_name: str = "gpt2"):
    prompts = prepare_dataset(num_samples=50, max_dataset_size=2000)
    good_reward = GoodRewardModel()
    hackable_reward = HackableRewardModel()
    print("\nTraining with GOOD reward model...")
    good_results = train_with_ppo(model_name, good_reward, prompts, num_epochs=2)
    print("\nTraining with HACKED reward model")
    hackable_results = train_with_ppo(model_name, hackable_reward, prompts, num_epochs=2)
    analyze_reward_hacking(good_results, hackable_results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reward Hacking Experiment with LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Examples: Qwen/Qwen2.5-0.5B"
    )

    args = parser.parse_args()
    main(model_name=args.model)
