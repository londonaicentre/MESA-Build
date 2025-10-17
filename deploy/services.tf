terraform {
  required_version = ">= 1.13.1"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

############################
# S3
############################

resource "aws_s3_bucket" "genollama_bucket" {
  bucket = "aicentrelondon-nlpteam-genollama"
}