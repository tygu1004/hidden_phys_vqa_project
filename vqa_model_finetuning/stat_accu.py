import json


def calculate_accuracy_by_answer(file_path):
    # 통계를 저장할 변수 초기화
    left_total = 0
    left_correct = 0
    right_total = 0
    right_correct = 0

    # JSONL 파일 읽기
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line.strip())
            answer = data.get("answer", "").strip()
            is_correct = data.get("is_correct", False)

            # 정답이 Left인 경우
            if answer == "Left.":
                left_total += 1
                if is_correct:
                    left_correct += 1
            # 정답이 Right인 경우
            elif answer == "Right.":
                right_total += 1
                if is_correct:
                    right_correct += 1

    # 정답률 계산 (0으로 나누는 에러 방지)
    left_accuracy = (left_correct / left_total * 100) if left_total > 0 else 0.0
    right_accuracy = (right_correct / right_total * 100) if right_total > 0 else 0.0

    total_total = left_total + right_total
    total_correct = left_correct + right_correct
    total_accuracy = (total_correct / total_total * 100) if total_total > 0 else 0.0

    # 결과 출력
    print(f"📊 정답 방향별 정확도 분석 결과")
    print("-" * 40)
    print(
        f"🟢 정답이 'Left.'인 경우 정답률 : {left_accuracy:.2f}% ({left_correct}/{left_total})"
    )
    print(
        f"🔵 정답이 'Right.'인 경우 정답률: {right_accuracy:.2f}% ({right_correct}/{right_total})"
    )
    print("-" * 40)
    print(f"전체 정답률: {total_accuracy:.2f}% ({total_correct}/{total_total})")


# 사용 예시 (실제 파일 이름으로 변경하여 실행하세요)
file_name = "outputs/intenvl_drop_spheres_rev_results.jsonl"
calculate_accuracy_by_answer(file_name)
