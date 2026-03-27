import json
import os

import numpy as np

# 병합된 결과 파일들이 들어있는 디렉토리
merged_dir = "merged_outputs/"


# 통계를 예쁘게 출력하기 위한 함수
def print_delta_accuracy(filename, stats):
    print(f"=== {filename} ===")

    # 구간 이름(구간 1, 2) 순서대로 정렬하여 출력
    for bin_name in sorted(stats.keys()):
        total = stats[bin_name]["total"]
        correct = stats[bin_name]["correct"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"  - {bin_name:<40} : {accuracy:>5.1f}% ({correct:>3}/{total:>3})")
    print("\n")


# 1. 첫 번째 순회: 태스크별로 모든 Delta 값을 수집하여 중앙값(Median) 계산
task_deltas = {"drop_spheres": [], "push_cubes": []}

if os.path.exists(merged_dir):
    for filename in os.listdir(merged_dir):
        if not filename.endswith(".jsonl"):
            continue

        is_mass_task = "push_cubes" in filename
        task_key = "push_cubes" if is_mass_task else "drop_spheres"

        filepath = os.path.join(merged_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                values = record.get("values", {})
                if len(values) >= 2:
                    val_list = list(values.values())
                    delta = abs(val_list[0] - val_list[1])
                    task_deltas[task_key].append(delta)

    # 태스크별 중앙값(50%) 계산
    task_thresholds = {}
    for task, deltas in task_deltas.items():
        if deltas:
            median_val = np.percentile(deltas, 50.0)
            task_thresholds[task] = median_val
            print(f"[{task}] 구간 임계값 자동 설정: 중앙값(50%) = {median_val:.3f}")
    print("\n" + "-" * 50 + "\n")

    # 2. 두 번째 순회: 계산된 중앙값을 바탕으로 정답률 계산
    for filename in sorted(os.listdir(merged_dir)):
        if not filename.endswith(".jsonl"):
            continue

        is_mass_task = "push_cubes" in filename
        task_key = "push_cubes" if is_mass_task else "drop_spheres"

        # 만약 해당 태스크의 데이터가 없어서 임계값이 세팅되지 않았다면 건너뜀
        if task_key not in task_thresholds:
            continue

        median_val = task_thresholds[task_key]
        delta_stats = {}
        filepath = os.path.join(merged_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                is_correct = record.get("is_correct", False)
                values = record.get("values", {})

                if len(values) >= 2:
                    val_list = list(values.values())
                    delta = abs(val_list[0] - val_list[1])

                    # 중앙값(Median)을 기준으로 2구간 분류
                    if delta <= median_val:
                        bin_name = f"1. Small Diff (<= {median_val:.3f}) [Hard]"
                    else:
                        bin_name = f"2. Large Diff (> {median_val:.3f}) [Easy]"

                    if bin_name not in delta_stats:
                        delta_stats[bin_name] = {"total": 0, "correct": 0}

                    delta_stats[bin_name]["total"] += 1
                    if is_correct:
                        delta_stats[bin_name]["correct"] += 1

        print_delta_accuracy(filename, delta_stats)
else:
    print(f"'{merged_dir}' 디렉토리를 찾을 수 없습니다.")
