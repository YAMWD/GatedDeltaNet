"""Thin adapters for the paper's Table 2 RULER configurations.

The sample generator and scorer remain those shipped by lm-eval.  The local
YAML files only add the 1K and 2K metric names that are absent from the
upstream task definition in lm-eval commit c1c4bea.
"""

from lm_eval.tasks.ruler.common_utils import (
    aggregate_metrics,
    process_results as upstream_process_results,
)
from lm_eval.tasks.ruler.niah_utils import (
    niah_single_1,
    niah_single_2,
    niah_single_3,
)


__all__ = [
    "aggregate_metrics",
    "process_results",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
]


def process_results(doc, results):
    """Expose every paper length while retaining the upstream score."""
    metrics = {str(length): -1.0 for length in (1024, 2048, 4096, 8192)}
    metrics.update(upstream_process_results(doc, results))
    return metrics
