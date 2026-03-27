import json
import os

import matplotlib.pyplot as plt

# 병합된 결과 파일들이 들어있는 디렉토리
merged_dir = "merged_outputs/"

# 중복 방지를 위해 고유한 파일명(비디오)을 기준으로 차이값을 저장할 딕셔너리
mass_deltas_dict = {}
restitution_deltas_dict = {}

# 디렉토리 내의 파일 순회
if os.path.exists(merged_dir):
    for filename in os.listdir(merged_dir):
        if not filename.endswith(".jsonl"):
            continue

        is_mass_task = "push_cubes" in filename
        filepath = os.path.join(merged_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                file_name = record.get("file_name")
                values = record.get("values", {})

                if file_name and len(values) >= 2:
                    val_list = list(values.values())
                    delta = abs(val_list[0] - val_list[1])

                    # 태스크별로 고유 비디오에 대한 Delta 값 저장
                    if is_mass_task:
                        mass_deltas_dict[file_name] = delta
                    else:
                        restitution_deltas_dict[file_name] = delta

    # 딕셔너리에서 값(Delta)들만 리스트로 추출
    mass_deltas = list(mass_deltas_dict.values())
    restitution_deltas = list(restitution_deltas_dict.values())

    # --- 그래프 그리기 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 탄성 차이(Restitution Delta) 분포 그래프
    axes[0].hist(
        restitution_deltas, bins=10, color="skyblue", edgecolor="black", alpha=0.7
    )
    axes[0].set_title(
        "Distribution of Restitution Differences (Drop Spheres)", fontsize=14
    )
    axes[0].set_xlabel("Difference in Restitution (0.0 ~ 1.0)", fontsize=12)
    axes[0].set_ylabel("Frequency (Number of Videos)", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # 2. 질량 차이(Mass Delta) 분포 그래프
    axes[1].hist(mass_deltas, bins=1, color="salmon", edgecolor="black", alpha=0.7)
    axes[1].set_title("Distribution of Mass Differences (Push Cubes)", fontsize=14)
    axes[1].set_xlabel("Difference in Mass (0.0 ~ 5.0)", fontsize=12)
    axes[1].set_ylabel("Frequency (Number of Videos)", fontsize=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    # 그래프를 이미지 파일로 저장
    plt.savefig("value_diff_distribution.png", dpi=300)
    print(
        "분포 그래프가 'value_diff_distribution.png' 파일로 성공적으로 저장되었습니다."
    )

else:
    print(f"'{merged_dir}' 디렉토리를 찾을 수 없습니다.")
