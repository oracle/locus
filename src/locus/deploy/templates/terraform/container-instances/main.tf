# OCI Container Instances: the simplest way to run the agent server on OCI.
# One container, one public or private IP, no cluster to operate. The
# container authenticates to OCI services with its resource principal.

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

data "oci_identity_availability_domains" "this" {
  compartment_id = var.compartment_id
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

locals {
  shape      = var.shape != null ? var.shape : (var.architecture == "arm64" ? "CI.Standard.A1.Flex" : "CI.Standard.E4.Flex")
  subnet_id  = var.create_network ? oci_core_subnet.public[0].id : var.subnet_id
  dns_label  = substr(replace(lower(var.name), "/[^a-z0-9]/", ""), 0, 15)
  registry   = regex("^([^/]+)/", var.image)[0]
  ad_name    = data.oci_identity_availability_domains.this.availability_domains[var.availability_domain_index].name
}

# ─── network (optional) ───────────────────────────────────────────────
# With create_network = true the root owns a VCN with one public subnet and an
# internet gateway, so a first deployment needs nothing in place: the instance
# pulls its image and reaches OCI services over its public IP. Production
# deployments usually set create_network = false and subnet_id to a subnet the
# network team owns, private, with a NAT gateway and a service gateway.

resource "oci_core_vcn" "this" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_id
  cidr_blocks    = [var.vcn_cidr]
  display_name   = "${var.name}-vcn"
  dns_label      = local.dns_label
}

resource "oci_core_internet_gateway" "this" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this[0].id
  display_name   = "${var.name}-igw"
}

resource "oci_core_route_table" "public" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this[0].id
  display_name   = "${var.name}-public-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.this[0].id
  }
}

resource "oci_core_security_list" "public" {
  count          = var.create_network ? 1 : 0
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.this[0].id
  display_name   = "${var.name}-public-sl"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  ingress_security_rules {
    source   = var.ingress_cidr
    protocol = "6"
    tcp_options {
      min = var.port
      max = var.port
    }
  }
}

resource "oci_core_subnet" "public" {
  count                      = var.create_network ? 1 : 0
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.this[0].id
  cidr_block                 = var.subnet_cidr
  display_name               = "${var.name}-public"
  dns_label                  = "pub"
  route_table_id             = oci_core_route_table.public[0].id
  security_list_ids          = [oci_core_security_list.public[0].id]
  prohibit_public_ip_on_vnic = false
}

# ─── the container instance ───────────────────────────────────────────

resource "oci_container_instances_container_instance" "this" {
  compartment_id                       = var.compartment_id
  availability_domain                  = local.ad_name
  display_name                         = var.name
  shape                                = local.shape
  container_restart_policy             = "ALWAYS"
  graceful_shutdown_timeout_in_seconds = 30

  shape_config {
    ocpus         = var.ocpus
    memory_in_gbs = var.memory_gbs
  }

  vnics {
    subnet_id             = local.subnet_id
    display_name          = "${var.name}-vnic"
    is_public_ip_assigned = var.assign_public_ip
  }

  containers {
    display_name          = "agent"
    image_url             = var.image
    command               = var.container_command
    arguments             = var.container_arguments
    environment_variables = module.environment.environment

    health_checks {
      health_check_type        = "HTTP"
      path                     = "/health"
      port                     = var.port
      interval_in_seconds      = 30
      timeout_in_seconds       = 5
      failure_threshold        = 3
      initial_delay_in_seconds = 20
    }
  }

  dynamic "image_pull_secrets" {
    for_each = var.image_pull_secret_id == null ? [] : [var.image_pull_secret_id]
    content {
      secret_type       = "VAULT"
      registry_endpoint = local.registry
      secret_id         = image_pull_secrets.value
    }
  }

  freeform_tags = var.freeform_tags
}

data "oci_core_vnic" "primary" {
  vnic_id = oci_container_instances_container_instance.this.vnics[0].vnic_id
}

output "container_instance_id" {
  value = oci_container_instances_container_instance.this.id
}

output "public_ip" {
  value       = data.oci_core_vnic.primary.public_ip_address
  description = "Public IP of the agent server, when assign_public_ip is true."
}

output "private_ip" {
  value = data.oci_core_vnic.primary.private_ip_address
}

output "url" {
  value       = "http://${coalesce(data.oci_core_vnic.primary.public_ip_address, data.oci_core_vnic.primary.private_ip_address)}:${var.port}"
  description = "Base URL of the agent server; /health, /invoke, /stream live under it."
}

output "resource_principal_rule" {
  value       = "ALL {resource.type = 'computecontainerinstance', resource.compartment.id = '${var.compartment_id}'}"
  description = "Matching rule for the dynamic group that grants the container its OCI permissions."
}
