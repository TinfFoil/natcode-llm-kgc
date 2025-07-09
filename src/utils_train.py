from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback
from transformers import TrainingArguments, BitsAndBytesConfig
import random
import torch

def get_quant_config(config):
    if config['load_in_4bit']:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    elif config['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    return quantization_config

class OutputPrinterCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, print_interval=10):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.print_interval = print_interval

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        if state.global_step % self.print_interval == 0:
            # Select a random sample from the dataset
            sample = random.choice(self.dataset)
            inputs = self.tokenizer(sample['text'], return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Perform a forward pass instead of generation
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get the most likely token at each step
            predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
            
            # Decode the predicted tokens
            generated_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
            
            print(f"\nStep {state.global_step} - Sample Output:")
            print(f"Input: {sample['text'][:100]}...")  # Print first 100 characters of input
            print(f"Model output: {generated_text}\n")

        return control

class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_message = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if key in logs}
            # print(log_message)
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print(state.global_step)
        return super().on_step_begin(args, state, control, **kwargs)
