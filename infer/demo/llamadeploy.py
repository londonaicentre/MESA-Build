import argparse
import os
import sys
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.huggingface import HuggingFaceModel, HuggingFacePredictor
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
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

def get_image_uri(region, instance_type):
    config = load_config()
    image_arguments = config["image_arguments"]
    return image_uris.retrieve(
        framework="huggingface",
        region=region,
        version=image_arguments["transformers_version"],
        base_framework_version=image_arguments["pytorch_version"],
        py_version=image_arguments["py_version"],
        instance_type=instance_type,
        image_scope="inference"
    )

def get_model(model_data, role, image_uri, region):
    return HuggingFaceModel(
        model_data=model_data,
        role=role,
        image_uri=image_uri,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
        model_server_workers=1
    )

def deploy_demo(model, instance_type, endpoint_name): 
    model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    return test_predict(endpoint_name)

def test_predict(endpoint_name):
    try:
        HuggingFacePredictor(endpoint_name=endpoint_name).predict(
            {
                "inputs": "hello world"
            }
        )
        return True
    except Exception as e:
        print(e)
        return False

def delete_demo(endpoint_name):
    predictor = HuggingFacePredictor(endpoint_name=endpoint_name)
    predictor.delete_model()
    predictor.delete_endpoint()

if __name__ == "__main__":
    args = parse_CLI_args()
    config = load_config()
    load_dotenv()

    if(args.command == "up"):
        if(deploy_demo(
            get_model(
                "s3://" + os.getenv("BUCKET") + "/" + args.path + "/model.tar.gz", 
                os.getenv("ROLE"), 
                get_image_uri(
                    config["llama"]["region"],
                    os.getenv("INSTANCE_TYPE")
                ),
                config["llama"]["region"]
            ),
            os.getenv("INSTANCE_TYPE"),
            config["llama"]["endpoint_name"]
        )):
            print("deployed")
        else:
            print("deploy failed")
    elif(args.command == "down"):
        delete_demo(config["llama"]["endpoint_name"])