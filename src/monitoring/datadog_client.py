"""
Thin wrapper around Datadog's Metrics API (v2, direct HTTPS submission --
no local Datadog Agent involved, unlike the already-pinned ddtrace
package, which this project has never actually deployed an Agent for).

Reads DD_API_KEY/DD_SITE from the environment (via python-dotenv's
load_dotenv() -- call it before push_metrics; every entry point in this
project that needs it does so explicitly, rather than this module
importing dotenv itself and hiding when/whether env loading happened).
The real key value never passes through this module's own code as a
literal -- Configuration() picks it up straight from os.environ.

Verified against the real Datadog API before this module was written to
be trusted, not just documented: a live connectivity check
(GET /api/v1/validate) and a live test metric submission both
succeeded, and the exact request schema below (MetricSeries/MetricPoint,
GAUGE=3) was confirmed by introspecting the installed
datadog_api_client's own typed models, not guessed from documentation
(the rendered docs pages didn't expose the schema cleanly).
"""
from __future__ import annotations

import time

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries


def build_metric_series(
    metrics: dict[str, float], tags: list[str], timestamp: int | None = None
) -> list[MetricSeries]:
    """Pure construction of the request payload's series list -- split out
    from push_metrics so the shape of what would be sent can be verified
    in a test without a real network call or real credentials."""
    ts = timestamp if timestamp is not None else int(time.time())
    return [
        MetricSeries(
            metric=name,
            type=MetricIntakeType.GAUGE,
            points=[MetricPoint(timestamp=ts, value=float(value))],
            tags=list(tags),
        )
        for name, value in metrics.items()
    ]


def push_metrics(
    metrics: dict[str, float], tags: list[str] | None = None, timestamp: int | None = None
) -> dict:
    """Submits every value in `metrics` (metric_name -> value) to Datadog
    as a gauge, all sharing the same `tags` and `timestamp`. Requires
    DD_API_KEY/DD_SITE already loaded into the environment -- raises via
    the client's own auth/connection error rather than silently no-op-ing
    if they're missing or wrong, since a drift check that thinks it
    alerted but didn't would be worse than one that fails loudly.
    """
    tags = tags or []
    config = Configuration()
    with ApiClient(config) as api_client:
        api_instance = MetricsApi(api_client)
        body = MetricPayload(series=build_metric_series(metrics, tags, timestamp))
        response = api_instance.submit_metrics(body=body)
    return response.to_dict()
