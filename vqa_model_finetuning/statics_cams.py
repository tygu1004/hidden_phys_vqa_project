import json
import os

# 병합된 결과 파일들이 들어있는 디렉토리
merged_dir = "outputs/"


# 결과를 예쁘게 출력하기 위한 함수
def print_camera_accuracy(filename, camera_stats):
    print(f"=== {filename} 결과 ===")

    # 정답률 기준으로 내림차순 정렬하여 출력
    sorted_stats = sorted(camera_stats.items())

    for cam_name, stats in sorted_stats:
        total = stats["total"]
        correct = stats["correct"]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"  - {cam_name:<20} : {accuracy:>5.1f}% ({correct:>2}/{total:>2})")
    print("\n")


# 1. 디렉토리 내의 모든 jsonl 파일 순회
if os.path.exists(merged_dir):
    for filename in sorted(os.listdir(merged_dir)):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(merged_dir, filename)

        # 카메라별 통계를 저장할 딕셔너리
        camera_stats = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                file_name = record.get("file_name", "")
                is_correct = record.get("is_correct", False)

                # 'episode_X_cam_front_narrow.mp4' 등에서 카메라 이름만 추출
                # 예: 'front_narrow', 'back_left_narrow' 등
                if "_cam_" in file_name:
                    cam_name = file_name.split("_cam_")[1].replace(".mp4", "")
                else:
                    cam_name = "unknown"

                # 딕셔너리에 초기화
                if cam_name not in camera_stats:
                    camera_stats[cam_name] = {"total": 0, "correct": 0}

                # 카운트 증가
                camera_stats[cam_name]["total"] += 1
                if is_correct:
                    camera_stats[cam_name]["correct"] += 1

        # 통계 출력
        print_camera_accuracy(filename, camera_stats)
else:
    print(f"'{merged_dir}' 디렉토리를 찾을 수 없습니다. 경로를 확인해주세요.")
