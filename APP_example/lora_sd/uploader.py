import gradio as gr
from huggingface_hub import HfApi


def upload(model_name: str, hf_token: str) -> None:
    api = HfApi(token=hf_token)
    user_name = api.whoami()["name"]
    model_id = f"{user_name}/{model_name}"
    try:
        api.create_repo(model_id, repo_type="model", private=True)
        api.upload_folder(repo_id=model_id, folder_path="results", path_in_repo="results", repo_type="model")
        url = f"https://huggingface.co/{model_id}"
        message = f"Your model was successfully uploaded to [{url}]({url})."
    except Exception as e:
        message = str(e)

    return gr.update(value=message, visible=True)
