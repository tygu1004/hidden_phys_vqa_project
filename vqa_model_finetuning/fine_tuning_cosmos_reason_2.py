import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Trainer,
    TrainingArguments,
)

data_files = {
    "train": [
        "mlcf-robot/push_cubes_vqa_dataset_r_train/metadata.jsonl",
        "mlcf-robot/push_cubes_vqa_dataset_l_train/metadata.jsonl",
        "mlcf-robot/drop_spheres_vqa_dataset_train/metadata.jsonl",
    ],
    "test": [
        "mlcf-robot/push_cubes_vqa_dataset_l_test/metadata.jsonl",
        "mlcf-robot/push_cubes_vqa_dataset_r_test/metadata.jsonl",
        "mlcf-robot/drop_spheres_vqa_dataset_test/metadata.jsonl",
    ],
}
dataset = load_dataset("json", data_files=data_files)
dataset = dataset.shuffle()
print(dataset)


class MultimodalDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.ignore_index = -100

    def __call__(self, features):
        system_prompt = "You are a helpful assistant. Answer with only a single word: 'left' or 'right'."

        full_messages_batch = []
        prompt_messages_batch = []

        for feature in features:
            full_message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": feature["file_name"],
                            "fps": 2.0,
                            "max_pixels": 224 * 224,
                        },
                        {"type": "text", "text": feature["question"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": feature["answer"]}],
                },
            ]
            full_messages_batch.append(full_message)

            prompt_message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": feature["file_name"],
                        },
                        {"type": "text", "text": feature["question"]},
                    ],
                },
            ]
            prompt_messages_batch.append(prompt_message)

        batch_inputs = self.processor.apply_chat_template(
            full_messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        labels = batch_inputs["input_ids"].clone()

        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index

        for i, prompt_msg in enumerate(prompt_messages_batch):
            prompt_enc = self.processor.apply_chat_template(
                [prompt_msg],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )

            prompt_len = len(prompt_enc["input_ids"][0])

            labels[i, :prompt_len] = self.ignore_index

        batch_inputs["labels"] = labels

        return batch_inputs


model_name = "nvidia/Cosmos-Reason2-8B"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16,
    attn_implementation="sdpa",
)
processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(model_name)

collate_function = MultimodalDataCollator(processor)

lora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,  # type of task to train on
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

cfg = TrainingArguments(
    output_dir="./cosmos_reason2_finetune_0325",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=0.1,
    num_train_epochs=3,
    ddp_find_unused_parameters=False,
    bf16=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="cosmos_reason2_finetune_0325",
    logging_steps=1,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collate_function,
    args=cfg,
)
trainer.train()
