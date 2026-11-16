variable "compartment_id" {
  type        = string
  description = "Compartment the container instance and its network live in."
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
  description = "Prefix for every resource this root creates."
}

variable "image" {
  type        = string
  description = "Full image reference, <region>.ocir.io/<namespace>/<repository>:<tag>."
}

variable "architecture" {
  type        = string
  default     = "amd64"
  description = "CPU architecture the image was built for; picks the shape family when shape is null."
  validation {
    condition     = contains(["amd64", "arm64"], var.architecture)
    error_message = "architecture must be amd64 or arm64."
  }
}

variable "shape" {
  type        = string
  default     = null
  description = "Container instance shape; null picks CI.Standard.E4.Flex (amd64) or CI.Standard.A1.Flex (arm64)."
}

variable "ocpus" {
  type        = number
  default     = 1
  description = "OCPUs for the instance. One serves a chat agent; raise it with concurrent tool work."
}

variable "memory_gbs" {
  type        = number
  default     = 4
  description = "Memory in GB. The SDK with the OCI provider idles near 300 MB; large contexts and RAG need more."
}

variable "port" {
  type        = number
  default     = 8080
  description = "Port the agent server listens on inside the container."
}

variable "container_command" {
  type        = list(string)
  default     = null
  description = "Override the image's entrypoint; null keeps the image's."
}

variable "container_arguments" {
  type        = list(string)
  default     = null
  description = "Override the image's command arguments; null keeps the image's."
}

variable "availability_domain_index" {
  type        = number
  default     = 0
  description = "Index into the compartment's availability domains."
}

variable "create_network" {
  type        = bool
  default     = true
  description = "Create a VCN with one public subnet; set false and pass subnet_id to use an existing one."
}

variable "subnet_id" {
  type        = string
  default     = null
  description = "Existing subnet for the instance when create_network is false."
}

variable "vcn_cidr" {
  type        = string
  default     = "10.90.0.0/16"
  description = "CIDR of the VCN this root creates."
}

variable "subnet_cidr" {
  type        = string
  default     = "10.90.1.0/24"
  description = "CIDR of the public subnet this root creates."
}

variable "ingress_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "Source CIDR allowed to reach the agent port. Narrow it to your callers."
}

variable "assign_public_ip" {
  type        = bool
  default     = true
  description = "Give the instance a public IP. False keeps it private behind a load balancer or VPN."
}

variable "image_pull_secret_id" {
  type        = string
  default     = null
  description = "Vault secret with the registry auth token, needed when the repository is private."
}

variable "server_api_key_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding LOCUS_SERVER_API_KEY, the bearer key clients send. Required with a public IP open to 0.0.0.0/0 unless allow_unauthenticated is true."
  validation {
    condition     = var.server_api_key_secret_id != null || var.allow_unauthenticated || !var.assign_public_ip || var.ingress_cidr != "0.0.0.0/0"
    error_message = "A public instance open to 0.0.0.0/0 is an agent anyone can call: set server_api_key_secret_id (the bearer key in Vault), or allow_unauthenticated = true if that is intended."
  }
}

variable "allow_unauthenticated" {
  type        = bool
  default     = false
  description = "Run the server without a bearer key; it refuses to start otherwise. Sets LOCUS_SERVER_ALLOW_UNAUTHENTICATED=1."
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
  description = "Extra environment variables for the agent server."
}

variable "freeform_tags" {
  type        = map(string)
  default     = {}
  description = "Freeform tags on the container instance."
}
