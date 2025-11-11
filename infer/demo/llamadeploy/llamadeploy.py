import argparse, os, sys, boto3, sagemaker
from sagemaker import image_uris
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor
from sagemaker.djl_inference import DJLModel, DJLPredictor
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.utils import load_config

def parse_CLI_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path within S3 bucket to the zipped weights of the model to deploy",
    )
    parser.add_argument(
        "command", 
        choices=["up", "down"]
    )
    arguments = parser.parse_args()
    return arguments

def get_image_uri(framework, region, instance_type):
    config = load_config()
    if framework not in list(config["images"].keys()):
        return None
    image_arguments = config["images"][framework]
    return image_uris.retrieve(
        framework=framework,
        region=region,
        version=image_arguments["version"],
        base_framework_version=image_arguments["pytorch_version"],
        py_version=image_arguments["py_version"],
        instance_type=instance_type,
        image_scope="inference"
    )

def get_model(model_data, role, image_uri, region):
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

def deploy_demo(model, instance_type, endpoint_name): 
    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    if isinstance(model, HuggingFaceModel):
        return test_predict(HuggingFacePredictor(endpoint_name=endpoint_name))
    elif isinstance(model, DJLModel):
        return test_predict(DJLPredictor(endpoint_name=endpoint_name))

def test_predict(predictor):
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

def delete_demo(image, endpoint_name):
    if "huggingface" in image:
        predictor = HuggingFacePredictor(endpoint_name=endpoint_name)
    elif "djl" in image:
        predictor = DJLPredictor(endpoint_name=endpoint_name)
    else:
        return False
    predictor.delete_model()
    predictor.delete_endpoint()
    return True

if __name__ == "__main__":
    args = parse_CLI_args()
    config = load_config()
    load_dotenv()

    if(args.command == "up"):
        print(
            deploy_demo(
                get_model(
                    "s3://" + os.getenv("BUCKET") + "/" + args.path + "/model.tar.gz", 
                    os.getenv("ROLE"), 
                    get_image_uri(
                        os.getenv("IMAGE"),
                        config["llama"]["region"],
                        os.getenv("INSTANCE_TYPE")
                    ),
                    config["llama"]["region"]
                ),
                os.getenv("INSTANCE_TYPE"),
                config["llama"]["endpoint_name"]
            )
        )
    elif(args.command == "down"):
        print(
            delete_demo(
                os.getenv("IMAGE"), 
                config["llama"]["endpoint_name"]
            )
        )