# The environment every target hands the agent server. One module, so the
# three roots agree on variable names and on what each one becomes inside the
# container. Secrets never appear here as values: a Vault secret OCID lands in
# <VAR>_SECRET_OCID and `python -m locus.deploy.entrypoint` resolves it into
# <VAR> at start-up with the instance or resource principal.

variable "compartment_id" {
  type        = string
  description = "Compartment the agent runs in; exported as OCI_COMPARTMENT_ID for the oci: model route."
}

variable "region" {
  type        = string
  description = "OCI region identifier; exported as OCI_REGION."
}

variable "auth_type" {
  type        = string
  description = "How the agent authenticates to OCI services: instance_principal or resource_principal."
  validation {
    condition     = contains(["instance_principal", "resource_principal"], var.auth_type)
    error_message = "auth_type must be instance_principal or resource_principal."
  }
}

variable "server_api_key_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding the bearer key clients send (LOCUS_SERVER_API_KEY)."
}

variable "oracle_dsn" {
  type        = string
  default     = null
  description = "Oracle AI Database connect string for the checkpointer (ORACLE_DSN). A one-way TLS string needs no wallet."
}

variable "oracle_user" {
  type        = string
  default     = null
  description = "Database user for the checkpointer (ORACLE_USER)."
}

variable "oracle_password_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding the database password (ORACLE_PASSWORD)."
}

variable "cache_url" {
  type        = string
  default     = null
  description = "OCI Cache (Redis protocol) endpoint, rediss://host:6379 (REDIS_URL)."
}

variable "object_storage_bucket" {
  type        = string
  default     = null
  description = "Object Storage bucket for checkpoints and audit anchors (LOCUS_BUCKET_NAME; LOCUS_OS_NAMESPACE is set with it)."
}

variable "object_storage_namespace" {
  type        = string
  default     = null
  description = "Object Storage namespace of the bucket."
}

variable "apm_data_upload_endpoint" {
  type        = string
  default     = null
  description = "OCI Application Performance Monitoring data-upload endpoint; sets the OTLP exporter variables."
}

variable "apm_data_key_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding the APM private data key; becomes the OTLP Authorization header."
}

variable "environment" {
  type        = map(string)
  default     = {}
  description = "Extra environment variables for the agent server; these win over the derived ones."
}

locals {
  derived = merge(
    {
      OCI_COMPARTMENT_ID  = var.compartment_id
      OCI_REGION          = var.region
      LOCUS_OCI_AUTH_TYPE = var.auth_type
    },
    var.server_api_key_secret_id == null ? {} : { LOCUS_SERVER_API_KEY_SECRET_OCID = var.server_api_key_secret_id },
    var.oracle_dsn == null ? {} : { ORACLE_DSN = var.oracle_dsn },
    var.oracle_user == null ? {} : { ORACLE_USER = var.oracle_user },
    var.oracle_password_secret_id == null ? {} : { ORACLE_PASSWORD_SECRET_OCID = var.oracle_password_secret_id },
    var.cache_url == null ? {} : { REDIS_URL = var.cache_url },
    var.object_storage_bucket == null ? {} : {
      LOCUS_BUCKET_NAME  = var.object_storage_bucket
      LOCUS_OS_NAMESPACE = var.object_storage_namespace
    },
    var.apm_data_upload_endpoint == null ? {} : {
      OTEL_EXPORTER_OTLP_ENDPOINT = "${var.apm_data_upload_endpoint}/20200101/opentelemetry/private"
      OTEL_EXPORTER_OTLP_PROTOCOL = "http/protobuf"
    },
    var.apm_data_key_secret_id == null ? {} : { LOCUS_APM_DATA_KEY_SECRET_OCID = var.apm_data_key_secret_id },
  )
}

output "environment" {
  value       = merge(local.derived, var.environment)
  description = "Environment variables for the agent server container."
}
