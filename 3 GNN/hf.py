from huggingface_hub import HfApi, HfFolder, upload_folder

upload_folder(
    folder_path="best_legal_bert_p_c_na",
    repo_id="suyamoonpathak/legalbert-pcna-finetuned",
    repo_type="model"
)
