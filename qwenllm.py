from xinference.client import RESTfulClient

def qwen_generate(base_url:str, model:str):

    size = float(model[8:-1])
    client = RESTfulClient(base_url)
    model_uid = client.launch_model(
        model_uid="my-qwen",
        model_name=model,
        model_format="pytorch",
        model_size_in_billions=size
    )

    return model_uid