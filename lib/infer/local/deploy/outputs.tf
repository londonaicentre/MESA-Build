output "api_key" {
  value     = aws_api_gateway_api_key.api_key.value
  sensitive = true
}

output "api_endpoint" {
  value = "https://${aws_api_gateway_rest_api.llama_s3_proxy.id}.execute-api.${var.aws_region}.amazonaws.com/${aws_api_gateway_stage.api_stage.stage_name}"
}