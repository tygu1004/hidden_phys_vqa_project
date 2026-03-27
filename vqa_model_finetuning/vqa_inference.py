import argparse
import json

import torch
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="VQA Inference with YAML Config")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # YAML 설정 로드
    config = load_config(args.config)

    model_type = config.get("model_type")
    model_name = config.get("model_name", "")
    meta_data_path = config.get("meta_data_path")
    output_path = config.get("output_path", "evaluation_results.jsonl")
    batch_size = config.get("batch_size", 16)
    finetuned = config.get("finetuned", False)
    peft_id = config.get("peft_id", "")

    print(f"--- Configuration ---")
    print(f"Model Type  : {model_type}")
    print(f"Finetuned   : {finetuned}")
    print(f"Data Path   : {meta_data_path}")
    print(f"Output Path : {output_path}")
    print(f"Batch Size  : {batch_size}")
    print(f"---------------------\n")

    # 1. 모델 및 프로세서 로드
    if model_type == "qwen":
        model_id = model_name or "Qwen/Qwen3.5-9B"
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

    elif model_type == "internvl":
        model_id = model_name or "OpenGVLab/InternVL3_5-8B-HF"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            size={"height": 448, "width": 448},
            do_sample_frames=True,
            fps=2.0,
        )

    elif model_type == "cosmos":
        model_id = model_name or "nvidia/Cosmos-Reason2-8B"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        raise ValueError(
            "지원하지 않는 model_type 입니다. 'qwen', 'internvl', 'cosmos' 중 하나를 선택하세요."
        )

    # PEFT (LoRA) 모델 적용
    if finetuned:
        from peft import PeftModelForQuestionAnswering

        if not peft_id:
            raise ValueError("에러: finetuned가 true일 경우 peft_id를 입력해야 합니다.")

        print(f"Loading PEFT weights from: {peft_id}...")
        model = PeftModelForQuestionAnswering.from_pretrained(
            model=model,
            model_id=peft_id,
            device_map="auto",
        )

    # 2. 메타데이터 로드
    meta_data = []
    with open(meta_data_path, "r") as f:
        for line in f:
            meta_data.append(json.loads(line))

    correct, wrong = 0, 0

    # 3. 인퍼런스 및 결과 저장
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(meta_data), batch_size)):
            batch_meta = meta_data[i : i + batch_size]
            batch_messages = []
            batch_answers = []

            # 배치 데이터 구성
            for meta in batch_meta:
                video_path = meta["file_name"]
                question = meta["question"]
                answer = meta["answer"]
                batch_answers.append(answer)

                msg = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a helpful assistant. Answer with only a single word: 'right' or 'left'.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video_path, "fps": 2.0},
                            {"type": "text", "text": f"{question}"},
                        ],
                    },
                ]
                if model_type == "qwen":
                    msg[1]["content"][0]["max_pixels"] = 640 * 480

                batch_messages.append(msg)

            # 4. Processor 적용
            inputs = processor.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
                fps=2.0,
                padding=True,
            )

            inputs = inputs.to(model.device)

            max_tokens = 1024
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # 6. 정답 비교 및 JSONL 기록
            for meta, out_text, ans in zip(batch_meta, output_texts, batch_answers):
                pred = out_text.strip()
                is_correct = ans.replace(".", "").lower() in pred.lower()

                if is_correct:
                    correct += 1
                else:
                    wrong += 1

                result_record = {
                    "file_name": meta["file_name"],
                    "question": meta["question"],
                    "answer": ans,
                    "prediction": pred,
                    "is_correct": is_correct,
                }
                out_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                out_f.flush()

    # 최종 결과 출력
    print("\n=== Evaluation Results ===")
    print(f"Results saved to: {output_path}")
    print(f"Correct: {correct}, Wrong: {wrong}")
    print(f"Accuracy: {correct / (correct + wrong):.4f}")


if __name__ == "__main__":
    main()
