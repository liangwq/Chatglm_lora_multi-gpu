import mii

#mii.serve("/root/autodl-tmp/GLM-API/model/chatglm3-6b-32k")
mii_configs = {"deployment_name":"mistral-deployment1"  }
model_config =  {"tensor_parallel":2,"model_name_or_path":"/root/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"
    }
client = mii.serve(
    	"/root/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62",
    deployment_name="mistral-deployment1",
    enable_restful_api=True,
    mii_config = mii_configs,
    model_config =model_config,
    restful_api_port=28080,)
