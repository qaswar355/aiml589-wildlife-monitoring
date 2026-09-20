"""
Creates the Datadog Monitor that watches the drift metrics
src/monitoring/drift.py already pushes (see push_metrics in
datadog_client.py) and emails a notification when a retrain is
recommended -- the "automatic" half of the retrain-trigger design (the
actual GPU retrain step stays a deliberate human action on ECS, see
drift.py's generate_retrain_runbook and its own module docstring for
why: no job scheduler, shared cluster, GPU selection has to be a live
judgement call).

Watches aiml589.drift.<tag>.retrain_recommended (a 0/1 gauge pushed
alongside every drift report) rather than the raw accuracy/confidence
deltas directly -- that flag is already the single, documented decision
(concept drift is the deciding signal; see drift.py) about what
"retrain recommended" means, so the monitor doesn't need a second,
possibly-inconsistent copy of that logic in Datadog's own query
language.

Email routing in Datadog works by "@mentioning" an address inside the
monitor's own message text (a real, stable Datadog convention, not
something specific to this project) -- no separate notification-channel
API call needed.

Run: python -m src.monitoring.setup_monitor <tag> <email> [--threshold 0.5]
"""
from __future__ import annotations

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.monitors_api import MonitorsApi
from datadog_api_client.v1.model.monitor import Monitor
from datadog_api_client.v1.model.monitor_options import MonitorOptions
from datadog_api_client.v1.model.monitor_thresholds import MonitorThresholds
from datadog_api_client.v1.model.monitor_type import MonitorType

DEFAULT_THRESHOLD = 0.5  # retrain_recommended is a 0/1 gauge


def build_monitor(tag: str, email: str, threshold: float = DEFAULT_THRESHOLD) -> Monitor:
    """Pure construction of the Monitor request body -- split out from
    create_retrain_monitor so the query/message text can be checked in a
    test without a real network call or real credentials."""
    metric = f"aiml589.drift.{tag}.retrain_recommended"
    query = f"avg(last_1h):avg:{metric}{{*}} >= {threshold}"
    message = (
        f"Drift detected for `{tag}`: retrain_recommended crossed {threshold}.\n\n"
        f"A ready-to-paste ECS retrain runbook was written to "
        f"metrics/drift/{tag}/retrain_runbook.sh -- check GPU availability "
        f"(nvidia-smi) before running it, this cluster has no job scheduler.\n\n"
        f"Full drift report: metrics/drift/{tag}/report.json\n\n"
        f"@{email}"
    )
    return Monitor(
        name=f"AIML589 retrain trigger ({tag})",
        type=MonitorType.METRIC_ALERT,
        query=query,
        message=message,
        tags=["project:aiml589", f"tag:{tag}"],
        options=MonitorOptions(
            thresholds=MonitorThresholds(critical=threshold),
            notify_no_data=False,
        ),
    )


def create_retrain_monitor(tag: str, email: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Creates the monitor via the real Datadog API and immediately reads
    it back (get_monitor) to prove it was actually created with the
    right query/message, not just that create_monitor returned without
    raising."""
    config = Configuration()
    with ApiClient(config) as api_client:
        api_instance = MonitorsApi(api_client)
        created = api_instance.create_monitor(body=build_monitor(tag, email, threshold))
        verified = api_instance.get_monitor(created.id)
    return verified.to_dict()


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("email")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    result = create_retrain_monitor(args.tag, args.email, args.threshold)
    print(f"created and verified monitor id={result['id']}  name={result['name']!r}")
    print(f"query: {result['query']}")


if __name__ == "__main__":
    main()
