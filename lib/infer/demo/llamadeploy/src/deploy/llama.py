import boto3
import sagemaker
from sagemaker import image_uris, Model, Predictor
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor
from sagemaker.djl_inference import DJLModel, DJLPredictor

from deploy.config import Config, ImageConfig

def get_image_uri(framework: str, region: str, instance_type: str) -> str | None:
    config: Config = Config()
    if framework not in list(config.images.keys()):
        return None
    image_arguments: ImageConfig = config.images[framework]
    return image_uris.retrieve(
        framework=framework,
        region=region,
        version=image_arguments.version,
        base_framework_version=image_arguments.pytorch_version,
        py_version=image_arguments.py_version,
        instance_type=instance_type,
        image_scope="inference"
    )

def get_model(model_data: str, role: str, image_uri: str, region: str) -> Model | None:
    if image_uri and 'huggingface' in image_uri:
        return HuggingFaceModel(
            model_data=model_data,
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
            model_server_workers=1
        )
    elif image_uri and 'djl' in image_uri:
        return DJLModel(
            model_data=model_data,
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
            env = {
                "OPTION_MAX_MODEL_LEN": "117128"
            }   
        )
    return None

def deploy_demo(model: Model, instance_type: str, endpoint_name: str) -> bool: 
    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    if isinstance(model, HuggingFaceModel):
        return test_predict(HuggingFacePredictor(endpoint_name=endpoint_name)) # type: ignore[no-untyped-call]
    elif isinstance(model, DJLModel):
        return test_predict(DJLPredictor(endpoint_name=endpoint_name)) # type: ignore[no-untyped-call]
    return False

def test_predict(predictor: Predictor) -> bool:
    try:
        predictor.predict(
            {
                "inputs": "hello world"
            }
        )
        return True
    except Exception as e:
        print(e)
        return False

def delete_demo(image: str, endpoint_name: str) -> bool:
    predictor: Predictor
    if "huggingface" in image:
        predictor = HuggingFacePredictor(endpoint_name=endpoint_name) # type: ignore[no-untyped-call]
    elif "djl" in image:
        predictor = DJLPredictor(endpoint_name=endpoint_name) # type: ignore[no-untyped-call]
    else:
        return False
    predictor.delete_model() # type: ignore[no-untyped-call]
    predictor.delete_endpoint() # type: ignore[no-untyped-call]
    return True

def run_deploy_up(bucket: str, path: str, sagemaker_execution_role: str, image: str, instance_type: str) -> bool:
    config: Config = Config()
    image_uri: str | None = get_image_uri(
        image,
        config.models["llama"].region,
        instance_type
    )
    if image_uri is None: return False
    model: Model | None = get_model(
        "s3://" + bucket + "/" + path + "/model.tar.gz", 
        sagemaker_execution_role, 
        image_uri,
        config.models["llama"].region
    )
    if model is None: return False
    return deploy_demo(
        model,
        instance_type,
        config.models["llama"].endpoint_name
    )

def run_deploy_down(image: str) -> bool:
    config: Config = Config()
    return delete_demo(
        image, 
        config.models["llama"].endpoint_name
    )