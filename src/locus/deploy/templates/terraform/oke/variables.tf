variable "compartment_id" {
  type        = string
  description = "Compartment the cluster lives in; also OCI_COMPARTMENT_ID for the agent."
}

variable "region" {
  type        = string
  description = "Region identifier, for example eu-frankfurt-1."
}

variable "oci_profile" {
  type        = string
  default     = null
  description = "Profile in ~/.oci/config for the provider and the kubeconfig exec plugin."
}

variable "name" {
  type        = string
  default     = "locus-agent"
  description = "Helm release name."
}

variable "namespace" {
  type        = string
  default     = "locus"
  description = "Kubernetes namespace for the release; created when missing."
}

variable "cluster_id" {
  type        = string
  default     = null
  description = "OKE cluster OCID; its kubeconfig is fetched from the API. Exclusive with kubeconfig_path."
}

variable "kubeconfig_path" {
  type        = string
  default     = null
  description = "Path to a kubeconfig for the cluster when cluster_id is null."
  validation {
    condition     = var.kubeconfig_path != null || var.cluster_id != null
    error_message = "Set cluster_id or kubeconfig_path."
  }
}

variable "image" {
  type        = string
  description = "Full image reference, <region>.ocir.io/<namespace>/<repository>:<tag>."
}

variable "architecture" {
  type        = string
  default     = "amd64"
  description = "CPU architecture of the image; pods are scheduled on nodes with kubernetes.io/arch equal to it."
  validation {
    condition     = contains(["amd64", "arm64"], var.architecture)
    error_message = "architecture must be amd64 or arm64."
  }
}

variable "image_pull_secret_name" {
  type        = string
  default     = null
  description = "Existing docker-registry Secret in the namespace, needed when the repository is private."
}

variable "workload_identity" {
  type        = bool
  default     = true
  description = "Use OKE workload identity (enhanced clusters; auth type oke_workload_identity). False uses the node's instance principal."
}

variable "replicas" {
  type        = number
  default     = 2
  description = "Replicas, and the floor for autoscaling."
}

variable "max_replicas" {
  type        = number
  default     = 2
  description = "Autoscaling ceiling; equal to replicas disables the autoscaler."
}

variable "port" {
  type        = number
  default     = 8080
  description = "Port the agent server listens on inside the container."
}

variable "service_type" {
  type        = string
  default     = "LoadBalancer"
  description = "LoadBalancer creates an OCI load balancer; ClusterIP for an ingress controller in front."
}

variable "load_balancer_min_mbps" {
  type        = number
  default     = 10
  description = "Flexible load balancer minimum bandwidth."
}

variable "load_balancer_max_mbps" {
  type        = number
  default     = 100
  description = "Flexible load balancer maximum bandwidth."
}

variable "cpu_request" {
  type    = string
  default = "500m"
}

variable "memory_request" {
  type    = string
  default = "1Gi"
}

variable "cpu_limit" {
  type    = string
  default = "2"
}

variable "memory_limit" {
  type    = string
  default = "4Gi"
}

variable "server_api_key_secret_id" {
  type        = string
  default     = null
  description = "Vault secret holding LOCUS_SERVER_API_KEY, the bearer key clients send. A Kubernetes Secret works too: serverApiKey.existingSecret in the chart."
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
