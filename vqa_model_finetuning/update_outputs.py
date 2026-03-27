import json
import os

output_dir = "outputs/"
prop_dir = "property_infos/"
merged_dir = "merged_outputs/"

# 병합된 결과를 저장할 새 디렉토리 생성
os.makedirs(merged_dir, exist_ok=True)

# 1. 파일 이름의 키워드와 테스트 셋 속성 파일(property) 매핑
task_mapping = {
    "drop_spheres": os.path.join(prop_dir, "drop_test_restitution.jsonl"),
    "push_cubes_l": os.path.join(prop_dir, "push_l_test_mass.jsonl"),
    "push_cubes_r": os.path.join(prop_dir, "push_r_test_mass.jsonl"),
}

# 2. 속성(property) 데이터를 Task별로 미리 읽어서 메모리에 로드
prop_data = {task: {} for task in task_mapping.keys()}

for task, filepath in task_mapping.items():
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if "file_name" in record and "values" in record:
                    prop_data[task][record["file_name"]] = record["values"]
    else:
        print(f"경고: {filepath} 파일이 존재하지 않습니다.")

# 3. outputs 디렉토리의 파일들을 순회하며 병합 진행
for filename in os.listdir(output_dir):
    if not filename.endswith(".jsonl"):
        continue

    # 현재 파일이 어떤 Task(키워드)에 속하는지 판별
    matched_task = None
    for task in task_mapping.keys():
        if task in filename:
            matched_task = task
            break

    # 매칭되는 Task가 있으면 데이터 병합 수행
    if matched_task:
        input_path = os.path.join(output_dir, filename)
        output_path = os.path.join(merged_dir, filename)

        with open(input_path, "r", encoding="utf-8") as fin, open(
            output_path, "w", encoding="utf-8"
        ) as fout:

            for line in fin:
                record = json.loads(line.strip())
                file_name = record.get("file_name")

                # property_infos에서 읽어온 데이터에 매칭되는 file_name이 있다면 values 추가
                if file_name in prop_data[matched_task]:
                    record["values"] = prop_data[matched_task][file_name]

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"처리 완료: {filename} -> {output_path}")

print("\n모든 테스트 셋 병합 작업이 완료되었습니다!")
