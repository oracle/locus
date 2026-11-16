# OKE: the agent server as a Helm release on an existing Kubernetes Engine
# cluster, for replicated, autoscaled production service. The root does not
# create the cluster; it reads the cluster's kubeconfig (cluster_id) or uses a
# kubeconfig file you already have (kubeconfig_path) and installs the bundled
# chart from ../../helm/locus-agent.

terraform {
  required_version = ">= 1.9"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 3.0"
    }
  }
}

provider "oci" {
  region              = var.region
  config_file_profile = var.oci_profile
}

data "oci_containerengine_cluster_kube_config" "this" {
  count         = var.cluster_id == null ? 0 : 1
  cluster_id    = var.cluster_id
  token_version = "2.0.0"
}

locals {
  kubeconfig = yamldecode(
    var.cluster_id == null ? file(var.kubeconfig_path) : data.oci_containerengine_cluster_kube_config.this[0].content
  )
  cluster = local.kubeconfig.clusters[0].cluster
  user    = local.kubeconfig.users[0].user
  exec    = try(local.user.exec, null)
  # Honour a profile the kubeconfig's exec plugin did not record.
  exec_args = local.exec == null ? null : (
    var.oci_profile == null || contains(local.exec.args, "--profile")
    ? local.exec.args
    : concat(local.exec.args, ["--profile", var.oci_profile])
  )
  image_parts = regex("^(.*):([^:/]+)$", var.image)
  auth_type   = var.workload_identity ? "oke_workload_identity" : "instance_principal"
  # The chart's fullname rule (templates/_helpers.tpl): the release name when it
  # already contains the chart name, else <release>-locus-agent, at most 63 chars.
  fullname = trimsuffix(substr(strcontains(var.name, "locus-agent") ? var.name : "${var.name}-locus-agent", 0, 63), "-")
}

provider "helm" {
  kubernetes = {
    host                   = local.cluster.server
    cluster_ca_certificate = base64decode(local.cluster["certificate-authority-data"])
    token                  = try(local.user.token, null)
    exec = local.exec == null ? null : {
      api_version = local.exec.apiVersion
      command     = local.exec.command
      args        = local.exec_args
    }
  }
}

module "environment" {
  source                    = "../modules/agent-environment"
  compartment_id            = var.compartment_id
  region                    = var.region
  auth_type                 = local.auth_type
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

resource "helm_release" "agent" {
  name             = var.name
  namespace        = var.namespace
  create_namespace = true
  chart            = "${path.module}/../../helm/locus-agent"
  timeout          = 600
  atomic           = true

  values = [yamlencode({
    replicaCount = var.replicas
    image = {
      repository = local.image_parts[0]
      tag        = local.image_parts[1]
      pullSecret = var.image_pull_secret_name
    }
    containerPort = var.port
    env           = module.environment.environment
    oci = {
      workloadIdentity = var.workload_identity
      region           = var.region
    }
    service = {
      type = var.service_type
      port = 80
      annotations = var.service_type == "LoadBalancer" ? {
        "oci.oraclecloud.com/load-balancer-type"                = "lb"
        "service.beta.kubernetes.io/oci-load-balancer-shape"    = "flexible"
        "service.beta.kubernetes.io/oci-load-balancer-shape-flex-min" = tostring(var.load_balancer_min_mbps)
        "service.beta.kubernetes.io/oci-load-balancer-shape-flex-max" = tostring(var.load_balancer_max_mbps)
      } : {}
    }
    resources = {
      requests = { cpu = var.cpu_request, memory = var.memory_request }
      limits   = { cpu = var.cpu_limit, memory = var.memory_limit }
    }
    autoscaling = {
      enabled                        = var.max_replicas > var.replicas
      minReplicas                    = var.replicas
      maxReplicas                    = var.max_replicas
      targetCPUUtilizationPercentage = 70
    }
    nodeSelector = { "kubernetes.io/arch" = var.architecture }
  })]
}

output "namespace" {
  value = helm_release.agent.namespace
}

output "release" {
  value = helm_release.agent.name
}

output "service" {
  value       = local.fullname
  description = "Service name; `kubectl -n <namespace> get svc <service>` shows the load balancer address."
}

output "service_account" {
  value       = local.fullname
  description = "ServiceAccount the chart creates; the workload-identity policy subject."
}

output "resource_principal_rule" {
  value       = var.workload_identity ? "workload identity: policy subject `request.principal.type = 'workload', request.principal.namespace = '${var.namespace}', request.principal.service_account = '${local.fullname}', request.principal.cluster_id = '${coalesce(var.cluster_id, "<cluster-id>")}'`" : "ALL {instance.compartment.id = '${var.compartment_id}'}"
  description = "What to put in the dynamic group (instance principal) or the policy (workload identity)."
}
