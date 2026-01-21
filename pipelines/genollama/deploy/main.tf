terraform {
  backend "s3" {
    bucket       = "aicentre-nlpteam-tfstate"
    key          = "schemallama/genollama/terraform.tfstate"
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

############################
# S3
############################

resource "aws_s3_bucket" "genollama_bucket" {
  bucket = "aicentre-nlpteam-genollama"
}