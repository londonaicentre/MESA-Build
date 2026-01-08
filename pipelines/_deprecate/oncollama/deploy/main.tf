terraform {
  backend "s3" {
    bucket       = "aicentre-nlpteam-tfstate"
    key          = "schemallama/oncollama/terraform.tfstate"
    region       = "eu-west-2"
    encrypt      = true
    use_lockfile = true
  }
  required_version = ">= 1.13.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = "eu-west-2"

  default_tags {
    tags = {
      terraform = "true"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  alias  = "us"

  default_tags {
    tags = {
      terraform = "true"
    }
  }
}

############################
# S3
############################

resource "aws_s3_bucket" "oncollama_bucket" {
  bucket = "aicentre-nlpteam-oncollama"
}

resource "aws_s3_bucket" "oncollama_us_bucket" {
  provider = aws.us
  bucket   = "aicentre-nlpteam-us-oncollama"
}

#####################################
# Local inference weights deployment
#####################################

module "infer_local_deploy" {
  source = "../../../lib/infer/local/deploy"
  bucket = aws_s3_bucket.oncollama_bucket.id
}

# Output

output "api_key" {
  value     = module.infer_local_deploy.api_key
  sensitive = true
}

output "api_endpoint" {
  value = module.infer_local_deploy.api_endpoint
}