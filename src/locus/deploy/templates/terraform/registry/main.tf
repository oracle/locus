# The OCIR repository the agent image is pushed to. `locus deploy apply` applies
# this root first, pushes the image, then applies the target root.

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

variable "compartment_id" {
  type        = string
  description = "Compartment that owns the repository."
}

variable "region" {
  type        = string
  description = "Region of the registry, for example eu-frankfurt-1."
}

variable "oci_profile" {
  type        = string
  default     = null
  description = "Profile in ~/.oci/config for the provider; null means DEFAULT or the OCI_CLI_* environment."
}

variable "name" {
  type        = string
  description = "Repository name; the image is <region>.ocir.io/<namespace>/<name>:<tag>."
}

variable "is_public" {
  type        = bool
  default     = false
  description = "A public repository needs no pull secret on the compute side; a private one needs an auth token in Vault."
}

data "oci_objectstorage_namespace" "this" {
  compartment_id = var.compartment_id
}

resource "oci_artifacts_container_repository" "this" {
  compartment_id = var.compartment_id
  display_name   = var.name
  is_public      = var.is_public
  is_immutable   = false
}

output "registry_host" {
  value       = "${var.region}.ocir.io"
  description = "Registry hostname for docker login and push."
}

output "image_repository" {
  value       = "${var.region}.ocir.io/${data.oci_objectstorage_namespace.this.namespace}/${oci_artifacts_container_repository.this.display_name}"
  description = "Repository path without a tag."
}

output "repository_id" {
  value = oci_artifacts_container_repository.this.id
}
