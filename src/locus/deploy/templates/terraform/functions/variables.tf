variable "compartment_id" {
  type        = string
  description = "Compartment the application and function live in."
}

variable "region" {
  type        = string
  description = "Region identifier, for example eu-frankfurt-1."
}

variable "oci_profile" {
  type        = string
  default     = null
  description = "Profile in ~/.oci/config for the provider; null means DEFAULT or the OCI_CLI_* environment."
}

variable "name" {
  type        = string
  default     = "locus-agent"
  description = "Application name; the function is <name>-server."
}

variable "image" {
  type        = string
  description = "Full image reference built from Dockerfile.functions, <region>.ocir.io/<namespace>/<repository>:<tag>."
}

variable "architecture" {
  type        = string
  default     = "amd64"
  description = "CPU architecture of the image: amd64 (GENERIC_X86) or arm64 (GENERIC_ARM)."
  validation {
    condition     = contains(["amd64", "arm64"], var.architecture)
    error_message = "architecture must be amd64 or arm64."
  }
}

variable "subnet_id" {
  type        = string
  description = "Subnet the application attaches to; a private subnet with a NAT or service gateway reaches OCI APIs."
}

variable "memory_mbs" {
  type        = number
  default     = 1024
  description = "Memory per invocation in MB (128 to 3072). The SDK with the OCI provider needs about 512 MB; 1024 leaves room for tools."
}

variable "timeout_seconds" {
  type        = number
  default     = 300
  description = "Invocation timeout in seconds, 300 at most; an agent turn with tools usually finishes in under 60."
}

variable "provisioned_concurrency" {
  type        = number
  default     = 0
  description = "Warm invocations kept ready to avoid cold starts; 0 disables it. Billed while provisioned."
}

variable "server_api_key_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding LOCUS_SERVER_API_KEY, the bearer key HTTP callers send."
}

variable "oracle_dsn" {
  type        = string
  default     = null
  description = "Oracle AI Database connect string for the checkpointer."
}

variable "oracle_user" {
  type        = string
  default     = null
  description = "Database user for the checkpointer."
}

variable "oracle_password_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding the database password."
}

variable "cache_url" {
  type        = string
  default     = null
  description = "OCI Cache endpoint, rediss://host:6379."
}

variable "object_storage_bucket" {
  type        = string
  default     = null
  description = "Object Storage bucket for checkpoints and audit anchors."
}

variable "object_storage_namespace" {
  type        = string
  default     = null
  description = "Object Storage namespace of the bucket."
}

variable "apm_data_upload_endpoint" {
  type        = string
  default     = null
  description = "OCI Application Performance Monitoring data-upload endpoint."
}

variable "apm_data_key_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding the APM private data key."
}

variable "environment" {
  type        = map(string)
  default     = {}
  description = "Extra configuration for the application; every function sees it as environment variables."
}

variable "freeform_tags" {
  type        = map(string)
  default     = {}
  description = "Freeform tags on the application and function."
}
