import boto3
import sagemaker
from sagemaker import image_uris, Model, Predictor
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor
from sagemaker.djl_inference import DJLModel, DJLPredictor

from deploy.config import Config, ImageConfig


def get_image_uri(framework: str, region: str, instance_type: str) -> str | None:
    """Get URI of inference container for SageMaker AI endpoint

    Args:
        framework (str): the base image to use. `djl-lmi` and `huggingface` are
            accepted values
        region (str): the region in which the container (endpoint) will be
            deployed
        instance_type (str): the type of instance in which to run the container

    Returns:
        str: The URI for use in endpoint deployment

    """
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
        image_scope="inference",
    )


def get_model(model_data: str, role: str, image_uri: str, region: str) -> Model | None:
    """Get wrapper model call for SageMaker AI endpoint deployment

    Args:
        model_data (str): Path in S3 bucket to model data
        role (str): The ARN of an IAM role with permissions to access SageMaker
        image_uri (str): The URI of the container to be used to serve the model
            within the endpoints
        region (str): The region in which to launch the endpoints

    Returns:
        Model: The model wrapper for endpoint deployment

    """
    if image_uri and "huggingface" in image_uri:
        return HuggingFaceModel(
            model_data=model_data,
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker.Session(
                boto_session=boto3.Session(region_name=region)
            ),
            model_server_workers=1,
        )
    elif image_uri and "djl" in image_uri:
        return DJLModel(
            model_data=model_data,
            role=role,
            image_uri=image_uri,
            sagemaker_session=sagemaker.Session(
                boto_session=boto3.Session(region_name=region)
            ),
            env={"OPTION_MAX_MODEL_LEN": "117128"},
        )
    return None


def deploy_demo(model: Model, instance_type: str, endpoint_name: str) -> bool:
    """Deploy a model to a SageMaker AI endpoint

    Args:
        model (Model): The wrapper model class to use for deployment
        instance_type (str): The type of instance to use in the endpoint
        endpoint_name (str): Name of the deployed endpoint

    Returns:
        bool: Whether the deployment was successful

    """
    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    if isinstance(model, HuggingFaceModel):
        return test_predict(HuggingFacePredictor(endpoint_name=endpoint_name))  # type: ignore[no-untyped-call]
    elif isinstance(model, DJLModel):
        return test_predict(DJLPredictor(endpoint_name=endpoint_name))  # type: ignore[no-untyped-call]
    return False


def test_predict(predictor: Predictor) -> bool:
    """Test deployed endpoint

    Args:
        predictor (Predictor): Wrapper prediction class for remote inference

    Returns:
        bool: Whether the endpoint is active

    """
    try:
        predictor.predict({"inputs": "hello world"})
        return True
    except Exception as e:
        print(e)
        return False


def delete_demo(image: str, endpoint_name: str) -> bool:
    """Remove a SageMaker AI endpoint

    Args:
        image (str): The type of image that has been deployed. `djl-lmi`
            and `huggingface` are accepted values.
        endpoint_name (str): The name of the deployed endpoint to delete

    Returns:
        bool: Whether the deletion was successful

    """
    predictor: Predictor
    if "huggingface" in image:
        predictor = HuggingFacePredictor(endpoint_name=endpoint_name)  # type: ignore[no-untyped-call]
    elif "djl" in image:
        predictor = DJLPredictor(endpoint_name=endpoint_name)  # type: ignore[no-untyped-call]
    else:
        return False
    predictor.delete_model()  # type: ignore[no-untyped-call]
    predictor.delete_endpoint()  # type: ignore[no-untyped-call]
    return True


def run_deploy_up(
    bucket: str,
    path: str,
    sagemaker_execution_role: str,
    image: str,
    instance_type: str,
) -> bool:
    """Deploy a model to a SageMaker AI endpoint

    Args:
        bucket (str): The name of the bucket where model weights are stored
        path (str): The path in the bucket to the weights
        sagemaker_execution_role (str): The ARN of an IAM role with
            permissions to access SageMaker
        image (str): the base image to use. `djl-lmi` and `huggingface` are
            accepted values
        instance_type (str): the type of instance in which to run the container

    Returns:
        bool: Whether the deployment was successful

    """
    config: Config = Config()
    image_uri: str | None = get_image_uri(
        image, config.models["llama"].region, instance_type
    )
    if image_uri is None:
        return False
    model: Model | None = get_model(
        "s3://" + bucket + "/" + path + "/model.tar.gz",
        sagemaker_execution_role,
        image_uri,
        config.models["llama"].region,
    )
    if model is None:
        return False
    return deploy_demo(model, instance_type, config.models["llama"].endpoint_name)


def run_deploy_down(image: str) -> bool:
    """Remove a SageMaker AI endpoint

    Args:
        image (str): The type of image that has been deployed. `djl-lmi`
            and `huggingface` are accepted values.

    Returns:
        bool: Whether the deletion was successful

    """
    config: Config = Config()
    return delete_demo(image, config.models["llama"].endpoint_name)
