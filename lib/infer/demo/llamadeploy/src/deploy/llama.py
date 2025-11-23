import boto3
import sagemaker
from sagemaker import image_uris, Model, Predictor
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor
from sagemaker.djl_inference import DJLModel, DJLPredictor

from deploy.config import Config, ImageConfig


class Deployer:
    """Deploy a Llama model on AWS SageMaker AI

    Args:
        image (str): The type of image that has been deployed. `djl-lmi`
            and `huggingface` are accepted values.
        instance_type (str): the type of instance in which to run the container

    """

    def __init__(self, image: str, instance_type: str):
        MODEL_KEY: str = "llama"
        self.__config: Config = Config()
        self.__image: str = image
        self.__instance_type: str = instance_type
        self.__model_region: str = self.__config.models[MODEL_KEY].region
        self.__model_endpoint_name: str = self.__config.models[MODEL_KEY].endpoint_name
        self.__model_max_length: int = self.__config.models[MODEL_KEY].max_length

    def __get_image_uri(self) -> str | None:
        """Get URI of inference container for SageMaker AI endpoint"""
        config: Config = Config()
        if self.__image not in list(config.images.keys()):
            return None
        image_arguments: ImageConfig = config.images[self.__image]
        return image_uris.retrieve(
            framework=self.__image,
            region=self.__model_region,
            version=image_arguments.version,
            base_framework_version=image_arguments.pytorch_version,
            py_version=image_arguments.py_version,
            instance_type=self.__instance_type,
            image_scope="inference",
        )

    def __get_model(self, model_data: str, role: str, image_uri: str) -> Model | None:
        """Get wrapper model call for SageMaker AI endpoint deployment

        Args:
            model_data (str): Path in S3 bucket to model data
            role (str): The ARN of an IAM role with permissions to access SageMaker
            image_uri (str): The URI of the container to be used to serve the model
                within the endpoints

        Returns:
            Model: The model wrapper for endpoint deployment

        """
        if image_uri and "huggingface" in image_uri:
            return HuggingFaceModel(
                model_data=model_data,
                role=role,
                image_uri=image_uri,
                sagemaker_session=sagemaker.Session(
                    boto_session=boto3.Session(region_name=self.__model_region)
                ),
                model_server_workers=1,
            )
        elif image_uri and "djl" in image_uri:
            return DJLModel(
                model_data=model_data,
                role=role,
                image_uri=image_uri,
                sagemaker_session=sagemaker.Session(
                    boto_session=boto3.Session(region_name=self.__model_region)
                ),
                env={"OPTION_MAX_MODEL_LEN": str(self.__model_max_length)},
            )
        return None

    def __deploy_demo(self, model: Model) -> bool:
        """Deploy a model to a SageMaker AI endpoint

        Args:
            model (Model): The wrapper model class to use for deployment

        Returns:
            bool: Whether the deployment was successful

        """
        model.deploy(
            initial_instance_count=1,
            instance_type=self.__instance_type,
            endpoint_name=self.__model_endpoint_name,
        )
        if isinstance(model, HuggingFaceModel):
            return self.__test_predict(
                HuggingFacePredictor(endpoint_name=self.__model_endpoint_name)  # type: ignore[no-untyped-call]
            )
        elif isinstance(model, DJLModel):
            return self.__test_predict(
                DJLPredictor(endpoint_name=self.__model_endpoint_name)  # type: ignore[no-untyped-call]
            )
        return False

    def __test_predict(self, predictor: Predictor) -> bool:
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

    def __delete_demo(self) -> bool:
        """Remove a SageMaker AI endpoint"""
        predictor: Predictor
        if "huggingface" in self.__image:
            predictor = HuggingFacePredictor(endpoint_name=self.__model_endpoint_name)  # type: ignore[no-untyped-call]
        elif "djl" in self.__image:
            predictor = DJLPredictor(endpoint_name=self.__model_endpoint_name)  # type: ignore[no-untyped-call]
        else:
            return False
        predictor.delete_model()  # type: ignore[no-untyped-call]
        predictor.delete_endpoint()  # type: ignore[no-untyped-call]
        return True

    def run_deploy_up(
        self,
        bucket: str,
        path: str,
        sagemaker_execution_role: str,
    ) -> bool:
        """Deploy a model to a SageMaker AI endpoint

        Args:
            bucket (str): The name of the bucket where model weights are stored
            path (str): The path in the bucket to the weights
            sagemaker_execution_role (str): The ARN of an IAM role with
                permissions to access SageMaker

        Returns:
            bool: Whether the deployment was successful

        """
        image_uri: str | None = self.__get_image_uri()
        if image_uri is None:
            return False
        model: Model | None = self.__get_model(
            "s3://" + bucket + "/" + path + "/model.tar.gz",
            sagemaker_execution_role,
            image_uri,
        )
        if model is None:
            return False
        return self.__deploy_demo(model)

    def run_deploy_down(self) -> bool:
        """Remove a SageMaker AI endpoint

        Returns:
            bool: Whether the deletion was successful

        """
        return self.__delete_demo()
