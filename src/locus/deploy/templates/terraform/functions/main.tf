# OCI Functions: the agent server as a function, for event-driven and bursty
# traffic. The image runs the Fn FDK with locus.deploy.functions translating
# each invocation into a request on the AgentServer ASGI app. The function
# authenticates to OCI services with its resource principal.

terraform {
  required_version = ">= 1.9"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0"
    }
  }
}

provider "oci" {
  region              = var.region
  config_file_profile = var.oci_profile
}

module "environment" {
  source                    = "../modules/agent-environment"
  compartment_id            = var.compartment_id
  region                    = var.region
  auth_type                 = "resource_principal"
  server_api_key_secret_id  = var.server_api_key_secret_id
  allow_unauthenticated     = var.allow_unauthenticated
  oracle_dsn                = var.oracle_dsn
  oracle_user               = var.oracle_user
  oracle_password_secret_id = var.oracle_password_secret_id
  cache_url                 = var.cache_url
  object_storage_bucket     = var.object_storage_bucket
  object_storage_namespace  = var.object_storage_namespace
  apm_data_upload_endpoint  = var.apm_data_upload_endpoint
  apm_data_key_secret_id    = var.apm_data_key_secret_id
  environment               = var.environment
}

resource "oci_functions_application" "this" {
  compartment_id = var.compartment_id
  display_name   = var.name
  subnet_ids     = [var.subnet_id]
  shape          = var.architecture == "arm64" ? "GENERIC_ARM" : "GENERIC_X86"
  config         = module.environment.environment
  freeform_tags  = var.freeform_tags
}

resource "oci_functions_function" "agent" {
  application_id     = oci_functions_application.this.id
  display_name       = "${var.name}-server"
  memory_in_mbs      = var.memory_mbs
  timeout_in_seconds = var.timeout_seconds
  freeform_tags      = var.freeform_tags

  source_details {
    source_type = "CONTAINER_IMAGE"
    image       = var.image
  }

  dynamic "provisioned_concurrency_config" {
    for_each = var.provisioned_concurrency == 0 ? [] : [var.provisioned_concurrency]
    content {
      strategy = "CONSTANT"
      count    = provisioned_concurrency_config.value
    }
  }
}

output "application_id" {
  value = oci_functions_application.this.id
}

output "function_id" {
  value = oci_functions_function.agent.id
}

output "invoke_endpoint" {
  value       = oci_functions_function.agent.invoke_endpoint
  description = "Invoke endpoint; `oci fn function invoke --function-id <function_id> --body '{\"prompt\": \"...\"}' --file -`."
}

output "resource_principal_rule" {
  value       = "ALL {resource.type = 'fnfunc', resource.compartment.id = '${var.compartment_id}'}"
  description = "Matching rule for the dynamic group that grants the function its OCI permissions."
}
