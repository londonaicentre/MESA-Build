############################
# API Gateway
############################

resource "aws_api_gateway_rest_api" "llama_s3_proxy" {
  name = "llama-s3-proxy"
  binary_media_types = [
    "application/zip"
  ]
}

# IAM

resource "aws_iam_role" "llama_s3_access_role" {
  name = "llamaS3AccessRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "apigateway.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "llama_s3_access_role_policy" {
  name = "llamaS3AccessRolePolicy"
  role = aws_iam_role.llama_s3_access_role.id

  policy = jsonencode({
  "Version": "2012-10-17",
  "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:ListBucket"
        ],
        "Resource": "arn:aws:s3:::${var.bucket}"
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject"
        ],
        "Resource": "arn:aws:s3:::${var.bucket}/*"
      }
    ]
  })
}

# Request

resource "aws_api_gateway_resource" "object" {
  rest_api_id = aws_api_gateway_rest_api.llama_s3_proxy.id
  parent_id   = aws_api_gateway_rest_api.llama_s3_proxy.root_resource_id
  path_part   = "{object+}"
}

resource "aws_api_gateway_method" "get" {
  rest_api_id      = aws_api_gateway_rest_api.llama_s3_proxy.id
  resource_id      = aws_api_gateway_resource.object.id
  http_method      = "GET"
  authorization    = "NONE"
  api_key_required = true

  request_parameters = {
    "method.request.path.object" = true
  }
}

resource "aws_api_gateway_integration" "llama_s3_proxy" {
  rest_api_id             = aws_api_gateway_rest_api.llama_s3_proxy.id
  resource_id             = aws_api_gateway_resource.object.id
  http_method             = aws_api_gateway_method.get.http_method
  integration_http_method = "GET"
  type                    = "AWS"
  uri                     = "arn:aws:apigateway:${var.aws_region}:s3:path/${var.bucket}/{object}"
  credentials             = aws_iam_role.llama_s3_access_role.arn

  request_parameters = {
    "integration.request.path.object" = "method.request.path.object"
  }

  depends_on = [aws_api_gateway_method.get]
}

# Response

resource "aws_api_gateway_method_response" "response_200" {
  rest_api_id = aws_api_gateway_rest_api.llama_s3_proxy.id
  resource_id = aws_api_gateway_resource.object.id
  http_method = aws_api_gateway_method.get.http_method
  status_code = "200"
}

resource "aws_api_gateway_integration_response" "llama_s3_proxy_response" {
  rest_api_id = aws_api_gateway_rest_api.llama_s3_proxy.id
  resource_id = aws_api_gateway_resource.object.id
  http_method = aws_api_gateway_method.get.http_method
  status_code = aws_api_gateway_method_response.response_200.status_code

  depends_on = [
    aws_api_gateway_integration.llama_s3_proxy
  ]
}

# Deployment

resource "aws_api_gateway_deployment" "llama_s3_proxy" {
  rest_api_id = aws_api_gateway_rest_api.llama_s3_proxy.id

  depends_on = [
    aws_api_gateway_integration.llama_s3_proxy
  ]
}

resource "aws_api_gateway_stage" "api_stage" {
  rest_api_id   = aws_api_gateway_rest_api.llama_s3_proxy.id
  deployment_id = aws_api_gateway_deployment.llama_s3_proxy.id
  stage_name    = "dist"
}

# API key

resource "aws_api_gateway_api_key" "api_key" {
  name = "s3-proxy-key"
}

resource "aws_api_gateway_usage_plan" "usage_plan" {
  name = "s3-proxy-plan"
  api_stages {
    api_id = aws_api_gateway_rest_api.llama_s3_proxy.id
    stage  = aws_api_gateway_stage.api_stage.stage_name
  }
}

resource "aws_api_gateway_usage_plan_key" "usage_plan_key" {
  key_id        = aws_api_gateway_api_key.api_key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.usage_plan.id
}

# Output

output "api_key" {
  value     = aws_api_gateway_api_key.api_key.value
  sensitive = true
}

output "api_endpoint" {
  value = "https://${aws_api_gateway_rest_api.llama_s3_proxy.id}.execute-api.${var.aws_region}.amazonaws.com/${aws_api_gateway_stage.api_stage.stage_name}"
}