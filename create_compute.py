"""
Azure ML Compute Cluster Creation Script.

This script defines and initiates the creation or update of an Azure Machine Learning
compute cluster. It requires the `azure-ai-ml` SDK and an authenticated `ml_client`
object to be available in the execution scope.

Example Usage (within a context where ml_client is defined):
    # Assuming ml_client is already configured and authenticated
    # (e.g., ml_client = MLClient(credential, subscription_id, resource_group, workspace_name))
    # Simply run this script or paste its content into a notebook cell.
"""
from azure.ai.ml.entities import AmlCompute
compute_name = "gpu-cluster"
cluster_basic = AmlCompute(
    name=compute_name,
    type="amlcompute",
    size="Standard_NC24s_v3",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
)
ml_client.begin_create_or_update(cluster_basic)