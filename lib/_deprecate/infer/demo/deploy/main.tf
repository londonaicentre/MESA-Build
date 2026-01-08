terraform {
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
# EC2
############################

# Roles
resource "aws_iam_role" "sagemaker_access_role" {
  name = "SagemakerAccessRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_secret_manager_policy" {
  role       = aws_iam_role.sagemaker_access_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "ec2-role-profile"
  role = aws_iam_role.sagemaker_access_role.name
}

# Security groups
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "ssh" {
  name        = "ssh"
  description = "ssh"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "litellm" {
  name        = "litellm"
  description = "litellm"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 4000
    to_port     = 4000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ui" {
  name        = "ui"
  description = "ui"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# SSH
resource "aws_key_pair" "keypair" {
  key_name   = "keypair"
  public_key = file("${var.keypair}.pub")
}

# Instance
data "aws_ssm_parameter" "ubuntu" {
  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

resource "aws_instance" "ec2_instance" {
  tags = {
    Name = "schemallama"
  }
  associate_public_ip_address = true
  instance_type               = "t3.small"
  ami                         = data.aws_ssm_parameter.ubuntu.value
  vpc_security_group_ids      = [aws_security_group.ssh.id, aws_security_group.litellm.id, aws_security_group.ui.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  key_name                    = aws_key_pair.keypair.key_name
  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    delete_on_termination = true
  }
}

############################
# Outputs
############################

output "keypair" {
  value = var.keypair
}

output "dns" {
  description = "dns"
  value       = aws_instance.ec2_instance.public_dns
}

resource "local_file" "env" {
  content  = "LITELLM_BASE_URL=${aws_instance.ec2_instance.public_dns}\nLITELLM_MASTER_KEY=${var.litellm_master_key}"
  filename = ".env"
}
