"""Four treatment methods sharing the production retrieval and scoring code."""
from scripts.experiment import gangmu
from scripts.experiment.dataset import PROJECT_ROOT, sha256_file

DISPLAY_NAME = "Gangmu ablation"
IMPLEMENTATION_DEPENDENCIES = (*gangmu.IMPLEMENTATION_DEPENDENCIES, "scripts/experiment/gangmu.py", "scripts/ablations/strategies.py", "scripts/ablations/binary_judge.py", "scripts/ablations/raw_resolver.py", "scripts/ablations/center_only_resolver.py")

def method_settings(dataset, config, options=None):
    options = dict(options or {})
    variant = options["variant"]
    settings = gangmu.method_settings(dataset, config, options)
    prefix = "raw_" if variant == "without_semantic_normalization" else "binary_" if variant == "binary_sev" else ""
    settings.update({"variant":variant, "alignment_prompt_sha256":sha256_file(PROJECT_ROOT/"prompts"/(prefix+"cross_doc_alignment.md")), "strict_prompt_sha256":sha256_file(PROJECT_ROOT/"prompts"/(prefix+"cross_doc_strict_cluster_pair.md")), "reranking_enabled":variant != "without_reranking", "representation":"statement" if variant == "without_semantic_normalization" else "normalized_statement", "representatives":"center_only" if variant == "center_only" else "full", "relation_space":"binary" if variant == "binary_sev" else "ternary"})
    return settings

run = gangmu.run
