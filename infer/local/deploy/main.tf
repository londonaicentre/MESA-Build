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

resource "aws_iam_role" "lambda_s3_access_role" {
  name = "LambdaS3AccessRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_s3_access_role_policy" {
  name = "LambdaS3AccessRolePolicy"
  role = aws_iam_role.lambda_s3_access_role.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:ListBucket"
        ],
        "Resource" : "arn:aws:s3:::${var.bucket}"
      },
      {
        "Effect" : "Allow",
        "Action" : [
          "s3:GetObject"
        ],
        "Resource" : "arn:aws:s3:::${var.bucket}/*"
      }
    ]
  })
}

# Lambda

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/llama_s3_presign.py"
  output_path = "${path.module}/llama_s3_presign.zip"
}


resource "aws_lambda_function" "presign_lambda" {
  filename      = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  function_name = "llama_s3_presign"
  role          = aws_iam_role.lambda_s3_access_role.arn
  handler       = "llama_s3_presign.handler"
  runtime       = "python3.12"

  environment {
    variables = {
      BUCKET_NAME = var.bucket
    }
  }
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
}

resource "aws_api_gateway_integration" "llama_s3_proxy" {
  rest_api_id             = aws_api_gateway_rest_api.llama_s3_proxy.id
  resource_id             = aws_api_gateway_resource.object.id
  http_method             = aws_api_gateway_method.get.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.presign_lambda.invoke_arn
  depends_on = [aws_api_gateway_method.get]
}

resource "aws_lambda_permission" "llama_s3_proxy" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.presign_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.llama_s3_proxy.execution_arn}/*/*"
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