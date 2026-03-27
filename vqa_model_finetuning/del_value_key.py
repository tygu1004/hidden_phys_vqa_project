import json
import os

input_file_path = ""
output_file_path = ""

with open(input_file_path, "r", encoding="utf-8") as infile, open(
    output_file_path, "w", encoding="utf-8"
) as outfile:

    for line in infile:
        data = json.loads(line)

        # 2. 'values' 키가 존재하면 삭제
        if "values" in data:
            del data["values"]

        # 3. 다시 JSON 문자열로 변환하여 새 파일에 쓰기 (ensure_ascii=False로 한글 깨짐 방지)
        modified_line = json.dumps(data, ensure_ascii=False)
        outfile.write(modified_line + "\n")

print(
    f"'{input_file_path}'에서 'values'를 삭제하고 '{output_file_path}'에 저장했습니다."
)
