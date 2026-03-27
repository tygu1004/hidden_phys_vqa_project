from huggingface_hub import HfApi

api = HfApi()

repo_id = "mlcf-robot/hidden_phys_vqa_datasets-v1"
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=True,
)

api.upload_large_folder(
    folder_path="/home/djjang/workspace/hidden_phys_vqa_project/vqa_model_finetuning/mlcf-robot",
    repo_id=repo_id,
    repo_type="dataset",
)
