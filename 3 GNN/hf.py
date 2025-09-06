from huggingface_hub import HfApi, HfFolder, upload_folder

upload_folder(
    folder_path="RoBERTa_prem_conc_finetuned",
    repo_id="suyamoonpathak/roberta-prem-conc-finetuned",
    repo_type="model"
)
