terraform {
  backend "s3" {
    bucket       = "aicentre-nlpteam-tfstate"
    key          = "mesa/build/terraform.tfstate"
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

resource "aws_s3_bucket" "mesa_build_bucket" {
  bucket = "aicentre-nlpteam-mesa-build"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "mesa_build_bucket_versioning" {
  bucket = aws_s3_bucket.mesa_build_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "terraform_state_bucket_lifecycle_configuration" {
  bucket = aws_s3_bucket.mesa_build_bucket.id

  rule {
    id = "remove_old_versions"
    noncurrent_version_expiration {
      newer_noncurrent_versions = 2
      noncurrent_days           = 1
    }

    status = "Enabled"
  }
}

resource "aws_s3_object" "mesa_build_bucket_documents" {
  bucket = aws_s3_bucket.mesa_build_bucket.id
  key    = "documents/"
}

resource "aws_s3_object" "mesa_build_bucket_trainingdata" {
  bucket = aws_s3_bucket.mesa_build_bucket.id
  key    = "trainingdata/"
}

resource "aws_s3_object" "mesa_build_bucket_models" {
  bucket = aws_s3_bucket.mesa_build_bucket.id
  key    = "models/"
}

resource "aws_s3_bucket" "mesa_public_bucket" {
  bucket = "aicentre-nlpteam-mesa-public"
}