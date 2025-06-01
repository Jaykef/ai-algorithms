import re
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoProcessor,
    AdamW,
    get_cosine_schedule_with_warmup
)
from transformers.image_utils import load_image
import difflib

# ==================================
# Implements Custom Multimodal GRPO Trainer That Scales for Small VLMs (SmolVLM-256M-Instruct)
# Paper: https://arxiv.org/abs/2402.03300
# Dataset: https://huggingface.co/datasets/hiyouga/geometry3k
# ==================================

# -------------------------------
# Reasoning System Prompt
# -------------------------------
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <thinking> </thinking><answer> answer here </answer>.
For example, always respond in this format:
<thinking>
Input our detailed chain-of-thought behind your answer goes here
</thinking>
<answer>
Input your final answer goes here, keep it short (if the final answer is number, only input the number here, no further explanation needed)
</answer>
"""

# -------------------------------
# Helper Functions
# -------------------------------
def extract_xml_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return text.strip()

def get_question(text: str) -> str:
    cleaned_text = re.sub(r'<image>|<end_of_utterance>', '', text)
    match = re.search(r'User:(.*?)System:', cleaned_text, re.DOTALL)
    return match.group(1).strip()

def get_assistant_response(text: str) -> str:
    if "Assistant:" in text:
        return text.split("Assistant:", 1)[1].strip()
    return text.strip()

def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<thinking>\n") == 1:
        count += 0.225
    if text.count("\n</thinking>\n") == 1:
        count += 0.225
    if text.count("\n<answer>\n") == 1:
        count += 0.225
        count -= len(text.split("\n</answer>")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.225
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def inference(prompt: str, image, model, tokenizer, processor, config, return_ids=False, use_template=True):
    if use_template:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SYSTEM_PROMPT + prompt}
                ]
            },
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            }
        ]
        full_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        full_prompt = prompt

    inputs = processor(text=full_prompt, images=[image], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_completion_length,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=config.temperature,
        num_return_sequences=config.num_generations, 
        use_cache=True
    )
    if return_ids:
        return outputs
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    if full_prompt in decoded:
        decoded = decoded.replace(full_prompt, "").strip()
    return decoded

# -------------------------------
# Reward Functions
# -------------------------------
def accuracy_reward(prompts, completions, answer, num_generated_samples_to_view=False, q_num=None, **kwargs) -> list:
    question = get_question(prompts[0])
    assistant_responses = get_assistant_response(completions[0])
    extracted_responses = extract_xml_answer(assistant_responses)
    image = kwargs.get("image", None)

    if num_generated_samples_to_view and q_num is not None:
        print(f"{'='*15} Completion {q_num} {'='*15}\nQuestion: {question}\n") 
        if image is not None:
            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        print(f"\nAnswer:\n{answer[0]}\n\nResponse:\n{assistant_responses}\n\nExtracted:\n{extracted_responses}\n\n{'='*18} End {'='*18}\n")

    reward_value = 2.0 if extracted_responses.strip() == answer[0].strip() else 0.0
    return [reward_value]

def xmlcount_reward(prompts, completions, answer, *args, **kwargs) -> list:
    return [count_xml(comp) for comp in completions]

def soft_format_reward(prompts, completions, answer, *args, **kwargs) -> list:
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    return [0.5 if re.match(pattern, comp, re.DOTALL) else 0.0 for comp in completions]

def strict_format_reward(prompts, completions, answer, *args, **kwargs) -> list:
    pattern = r"^<thinking>\n.*?\n</thinking>\n<answer>\n.*?\n</answer>\n$"
    return [0.5 if re.match(pattern, comp) else 0.0 for comp in completions]

def int_reward(prompts, completions, answer, *args, **kwargs) -> list:
    return [0.5 if get_assistant_response(comp).isdigit() else 0.0 for comp in completions]

def sequence_similarity_reward(prompts, completions, answer, *args, **kwargs) -> list:
    extracted_responses = [extract_xml_answer(get_assistant_response(r)) for r in completions]
    rewards = []
    for pred_patch, gt_patch in zip(extracted_responses, [answer] * len(completions)):
        if not pred_patch:
            rewards.append(-1.0)
        else:
            matcher = difflib.SequenceMatcher(None, pred_patch, gt_patch[0])
            similarity = matcher.ratio()
            rewards.append(similarity)
    return rewards

def clean_text(text):
    return re.sub(r"<image>", "", text).strip()

# -------------------------------
# Get & Prep data with Chat Template
# -------------------------------
def prep_geometry3k_data(split="train") -> Dataset:
    data = load_dataset('hiyouga/geometry3k', split=split)
    data = data.map(lambda x: {
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": [load_image((x["images"][0]))]},
                    {"type": "text", "text": clean_text(x["problem"])}
                ]
            },
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            }
        ],
        "answer": x["answer"]
    }, num_proc=8, batched=False)
    return data

dataset = prep_geometry3k_data()

# -------------------------------
# Custom GRPO Config
# -------------------------------
class GRPOConfig:
    def __init__(self, **kwargs):
        self.output_dir = kwargs.get("output_dir", "outputs")
        self.run_name = kwargs.get("run_name", "custom_grpo")
        self.learning_rate = kwargs.get("learning_rate", 1e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.warmup_steps = kwargs.get("warmup_steps", 50)
        self.num_generations = kwargs.get("num_generations", 1)
        self.max_prompt_length = kwargs.get("max_prompt_length", 2048)
        self.max_completion_length = kwargs.get("max_completion_length", 100)
        self.num_train_epochs = kwargs.get("num_train_epochs", 1)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.clip_epsilon = kwargs.get("clip_epsilon", 0.2)
        self.beta = kwargs.get("beta", 0.01)
        self.logging_steps = kwargs.get("logging_steps", 1)
        self.save_steps = kwargs.get("save_steps", 50)
        self.max_steps = kwargs.get("max_steps", 40)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = kwargs.get("temperature", 0.2)
        self.num_generated_samples_to_view = kwargs.get("num_generated_samples_to_view", 5)
        self.bf16 = kwargs.get("bf16", False)
        self.use_vllm = kwargs.get("use_vllm", False)
        self.vllm_device = kwargs.get("vllm_device", "auto")
        self.vllm_gpu_memory_utilization = kwargs.get("vllm_gpu_memory_utilization", 0.2)
        self.vllm_dtype = kwargs.get("vllm_dtype", "float16")
        self.vllm_max_model_len = kwargs.get("vllm_max_model_len", 2048)
        self.vllm_enforce_eager = kwargs.get("vllm_enforce_eager", False)
        self.sync_ref_model = True


# -------------------------------
# Custom GRPO Trainer
# -------------------------------
class GRPOTrainer:
    def __init__(self, model, tokenizer, reward_funcs, config, train_dataset, processor):
        self.dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_funcs = reward_funcs
        self.config = config
        self.train_dataset = train_dataset

        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = (len(train_dataset) // config.gradient_accumulation_steps) * config.num_train_epochs
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=config.warmup_steps,
                                                         num_training_steps=total_steps)

        self.ref_model = AutoModelForVision2Seq.from_pretrained(model.config._name_or_path, torch_dtype=torch.float16) 
        self.ref_model.to(config.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.step = 0
        self._metrics = defaultdict(list)
        self.scaler = torch.amp.GradScaler(enabled=config.bf16 and config.device.startswith("cuda")) if config.device.startswith("cuda") else None

        if self.config.use_vllm:
            from vllm import LLM, SamplingParams
            self.llm = self._initialize_vllm()
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_completion_length,
                n=self.config.num_generations,
                stop_token_ids=[self.tokenizer.eos_token_id]
            )
        else:
            self.llm = None
            self.sampling_params = None

    def clear_memory_cache(self):
        if self.config.device.startswith("cuda"):
            with torch.no_grad():
                torch.cuda.empty_cache()

    def sync_ref_model(self):
        if self.step % 25 == 0 and self.step > 0:
            self.ref_model.load_state_dict(self.model.state_dict())
            # print(f"Reference model synchronized at step {self.step}")

    def _initialize_vllm(self):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM is not available. Please install vllm with `pip install vllm`.")

        vllm_device = self.config.vllm_device
        if vllm_device == "auto":
            if torch.cuda.device_count() == 1:
                vllm_device = "cuda:0"
            else:
                vllm_device = f"cuda:{self.config.device}"

        if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
            raise ValueError(f"The requested device for vLLM ({vllm_device}) is not available.")

        llm_instance = LLM(
            model=self.model.config._name_or_path,
            device=vllm_device,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            dtype=self.config.vllm_dtype,
            max_model_len=self.config.vllm_max_model_len,
            enable_prefix_caching=True,
            enforce_eager=self.config.vllm_enforce_eager
        )
        return llm_instance

    def get_per_token_logps(self, model, full_ids, num_logits_to_keep):
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :] 
        if num_logits_to_keep <= 0:
            return torch.empty(0, dtype=torch.float32, device=full_ids.device)
        logits_slice = logits[:, -num_logits_to_keep:, :]
        token_ids = full_ids[:, -num_logits_to_keep:]
        log_probs = torch.log_softmax(logits_slice, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs

    def compute_loss(self, input_ids, generation_output, advantages):
        num_logits_to_keep = generation_output.shape[1] - input_ids.shape[1]
        full_ids = generation_output

        if num_logits_to_keep <= 0:
            return torch.tensor(0.0, device=input_ids.device), 0.0, 0.0

        per_token_logps = self.get_per_token_logps(self.model, full_ids, num_logits_to_keep)
        with torch.no_grad():
            ref_per_token_logps = self.get_per_token_logps(self.ref_model, full_ids, num_logits_to_keep)

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        per_token_kl = torch.clamp(per_token_kl, max=10) 
        device = input_ids.device
        completion_ids = full_ids[:, input_ids.shape[1]:]
        is_eos = (completion_ids == self.tokenizer.eos_token_id)
        batch_size, seq_len = is_eos.size()
        eos_idx = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        for i in range(batch_size):
            nonzero = torch.nonzero(is_eos[i], as_tuple=False)
            if nonzero.numel() > 0:
                eos_idx[i] = nonzero[0, 0]
        sequence_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (sequence_indices <= eos_idx.unsqueeze(1)).float()

        ratios = torch.exp(per_token_logps - ref_per_token_logps)
        clipped_ratios = ratios.clamp(1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
        advantages_expanded = advantages.unsqueeze(1).expand_as(ratios)
        token_surrogate = torch.min(ratios * advantages_expanded, clipped_ratios * advantages_expanded)
        surrogate_per_example = (token_surrogate * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8) # Added small epsilon for stability
        kl_per_example = (per_token_kl * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8) # Added small epsilon for stability
        J_per_example = surrogate_per_example - self.config.beta * kl_per_example
        loss = J_per_example.mean()

        mean_kl = kl_per_example.mean().item()
        completion_length = mask.sum(dim=1).mean().item()
        return loss, mean_kl, completion_length

    def evaluate_rewards(self, prompt, completions, gt_answer, image):
        rewards_dict = {}
        combined_rewards = []
        for i, comp in enumerate(completions):
            sample_rewards = []
            for func in self.reward_funcs:
                r = func([prompt], [comp], [gt_answer], image=image, **self.config.__dict__)
                sample_rewards.append(r[0])
                rewards_dict.setdefault(func.__name__, []).append(r[0])
            combined_reward = sum(sample_rewards)
            combined_rewards.append(combined_reward)
        return combined_rewards, rewards_dict

    def train(self):
        self.model.train()

        for epoch in range(self.config.num_train_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                if self.step >= self.config.max_steps:
                    break

                self.clear_memory_cache()
                example = batch[0]
                messages = example["prompt"]
                gt_answer = example["answer"]
                image = load_image(example["images"][0])

                prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = self.processor(text=prompt_text, images=[image],
                                        return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                with torch.autocast(
                    device_type=self.config.device,
                    enabled=(self.scaler is not None),
                    dtype=(torch.float16 if not self.config.bf16 else torch.bfloat16)
                ):
                    if self.config.use_vllm:
                        vllm_prompts = [prompt_text] * self.config.num_generations
                        vllm_inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for prompt in vllm_prompts]
                        vllm_outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False)
                        completions = [output.outputs[0].text for output in vllm_outputs]

                        generation_output_ids = []
                        for completion_text in completions:
                            full_text = prompt_text + completion_text
                            tokenized_output = self.processor(text=full_text, images=[image], return_tensors="pt", add_special_tokens=False)
                            gen_ids = tokenized_output["input_ids"][:, inputs["input_ids"].shape[1]:].to(self.config.device)
                            if gen_ids.numel() == 0:
                                print(f"Warning: vLLM generated no new tokens, step {self.step}. Completion likely stopped early.")
                                continue
                            generation_output_ids.append(gen_ids)

                        if not generation_output_ids:
                            print(f"Warning: No valid generation outputs after processing vLLM outputs, step {self.step}. Skipping step.")
                            continue
                        else:
                            max_gen_len = max(gen_ids.shape[1] for gen_ids in generation_output_ids)
                            padded_generation_output_ids = [
                                F.pad(gen_ids, (0, max_gen_len - gen_ids.shape[1]), 'constant', self.tokenizer.pad_token_id)
                                for gen_ids in generation_output_ids
                            ]
                            generation_output_ids = torch.cat(padded_generation_output_ids, dim=0)
                            inputs["input_ids"] = inputs["input_ids"].to(self.config.device)
                            generation_output = torch.cat([inputs["input_ids"].repeat(self.config.num_generations, 1), generation_output_ids], dim=1)

                    else:
                        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                        inputs = self.processor(text=prompt_text, images=[load_image(example["images"][0])],
                                                return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                        generation_output = inference(prompt_text, load_image(example["images"][0]),
                                                    self.model, self.tokenizer, self.processor, self.config,
                                                    return_ids=True, use_template=False)

                    completions = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generation_output]
                    completions = [c.replace(prompt_text, "").strip() if prompt_text in c else c for c in completions]

                    num_gens = len(completions)
                    view_flag = (self.step < self.config.num_generated_samples_to_view)

                    acc_rewards = accuracy_reward([prompt_text] * num_gens, completions, [gt_answer] * num_gens,
                                num_generated_samples_to_view=view_flag, q_num=self.step, image=load_image(example["images"][0]))

                    seq_sim_rewards = sequence_similarity_reward([prompt_text]*num_gens, completions, [gt_answer]*num_gens)
                    combined_rewards, reward_dict = self.evaluate_rewards(prompt_text, completions, gt_answer, image)
                    rewards_tensor = torch.tensor(combined_rewards, device=self.config.device, dtype=torch.float)

                    reward_avg = rewards_tensor.mean().item()
                    reward_std = rewards_tensor.std().item() if rewards_tensor.numel() > 1 else 0.0

                    if self.config.num_generations > 1:
                        rewards_grouped = rewards_tensor.view(-1, self.config.num_generations)
                        mean_rewards = rewards_grouped.mean(dim=1)
                        std_rewards = rewards_grouped.std(dim=1) + 1e-8
                        advantages = (rewards_tensor - mean_rewards.repeat_interleave(self.config.num_generations)) / std_rewards.repeat_interleave(self.config.num_generations)
                    else:
                        advantages = rewards_tensor
                    advantages = torch.clamp(advantages, -5.0, 5.0)

                    num_logits_to_keep = generation_output.shape[1] - inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        old_logps = self.get_per_token_logps(self.ref_model, generation_output[:, :inputs["input_ids"].shape[1] + max(0, num_logits_to_keep)], max(0, num_logits_to_keep)).detach()

                    loss, mean_kl, completion_length = self.compute_loss(inputs["input_ids"].repeat(self.config.num_generations, 1), generation_output[:, :inputs["input_ids"].shape[1] + max(0, num_logits_to_keep)], advantages)

                    loss = loss / self.config.gradient_accumulation_steps
                    if self.scaler is not None and not torch.isnan(loss):
                        self.scaler.scale(loss).backward()
                    elif not torch.isnan(loss): 
                        loss.backward()
                    else:
                        print("Warning: Loss is NaN, skipping backward pass.")


                    if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
                        self.clear_memory_cache()
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                            self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        self.step += 1 
                        self.sync_ref_model()

                        self._metrics["loss"].append(loss.item() * self.config.gradient_accumulation_steps)
                        self._metrics["completion_length"].append(completion_length)
                        self._metrics["reward"].append(reward_avg)
                        self._metrics["reward_std"].append(reward_std)
                        self._metrics["accuracy_reward"].append(sum(acc_rewards) / len(acc_rewards) if acc_rewards else 0)
                        self._metrics["sequence_similarity_reward"].append(sum(seq_sim_rewards) / len(seq_sim_rewards) if seq_sim_rewards else 0)
                        self._metrics["kl"].append(mean_kl)
                        print(f"Step {self.step} | Loss: {loss.item()*self.config.gradient_accumulation_steps:.4f} | "
                            f"Reward: {reward_avg:.4f} | Reward Std: {reward_std:.4f} | "
                            f"Completion Length: {completion_length:.4f} | KL: {mean_kl:.4f}\n")

                if self.step >= self.config.max_steps:
                    break
            if self.step >= self.config.max_steps:
                break

        final_model_path = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        self.model.save_pretrained(final_model_path)
        self.processor.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")

        plt.figure(figsize=(14, 10))
        plt.subplot(3, 2, 1)
        plt.plot(self._metrics["reward"], label="Reward", color="green")
        plt.title("Reward vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(self._metrics["completion_length"], label="Avg Completion Length", color="purple")
        plt.title("Avg Completion Length vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Completion Length")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(self._metrics["accuracy_reward"], label="Accuracy", color="blue")
        plt.title("Accuracy vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(self._metrics["reward_std"], label="Reward Std", color="orange")
        plt.title("Reward Std vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Reward Std")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(self._metrics["kl"], label="KL Penalty", color="red")
        plt.title("KL Penalty vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("KL Penalty")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(self._metrics["sequence_similarity_reward"], label="Sequence Similarity", color="cyan")
        plt.title("Sequence Similarity vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Sequence Similarity")
        plt.legend()


        plt.tight_layout()
        plt.show()

# -------------------------------
# Load Model
# -------------------------------
model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
processor.image_processor.size = {"height": 512, "width": 512}
# print("Max image size:", processor.image_processor.size)

model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# Config & Train
# -------------------------------
config = GRPOConfig(
    output_dir="outputs/smolvl_grpo_geometry3k_vllm",
    run_name="smolvl_256m_grpo_geometry3k_reasoner_vllm",
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_steps=20,
    num_generations=2,
    max_prompt_length=2048,
    max_completion_length=150,
    num_train_epochs=1,
    gradient_accumulation_steps=1,
    clip_epsilon=0.2,
    beta=0.01,
    logging_steps=1,
    save_steps=20,
    max_steps=20,
    temperature=0.2,
    num_generated_samples_to_view=15,
    bf16=True,
    # use_vllm=True,
    # vllm_device="auto",
    # vllm_gpu_memory_utilization=0.2,
    # vllm_dtype="float16",
    # vllm_max_model_len=2048,
    # vllm_enforce_eager=False,
)

reward_functions = [xmlcount_reward, soft_format_reward, strict_format_reward, int_reward, accuracy_reward, sequence_similarity_reward]
trainer = GRPOTrainer(model, tokenizer, reward_functions, config, dataset, processor)
trainer.train()

# -------------------------------
# Sample from final
# -------------------------------
sample_image = "https://raw.githubusercontent.com/Jaykef/ai-algorithms/refs/heads/main/assets/sample.jpg"
if config.use_vllm:
    def inference_vllm(prompt: str, image, trainer, config):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            }
        ]
        full_prompt = trainer.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print(f"\nSample From Trained Model:\n")
        plt.figure()
        plt.imshow(load_image(sample_image))
        plt.axis('off')
        plt.show()

        vllm_input = {"prompt": full_prompt, "multi_modal_data": {"image": load_image(sample_image)}}
        vllm_outputs = trainer.llm.generate(vllm_input, sampling_params=trainer.sampling_params, use_tqdm=False)
        final_inference_result = [output.outputs[0].text for output in vllm_outputs]
        return final_inference_result[0]

    print(inference_vllm("Which of the given answers A, B, C, D in the image is correct?\n", sample_image, trainer, config))
else:
    print(f"\nSample From Trained Model:\n")
    plt.figure()
    plt.imshow(load_image(sample_image))
    plt.axis('off')
    plt.show()
    assistant_response = inference("Which of the given answers A, B, C, D in the image is correct?", sample_image, 
                    model, tokenizer, processor, config, return_ids=False)
    extracted_answer = get_assistant_response(assistant_response) 
    print(extracted_answer) 
    