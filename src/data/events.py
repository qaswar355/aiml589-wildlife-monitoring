"""
Burst-event reconstruction for camera trap sequences.

There is no `seq_id` in the LILA NZ trail-cam metadata, but cameras still
fire in bursts of several near-identical frames per real animal encounter.
Naively per-species-capping raw frames (e.g. 8,000 mouse frames) can mean
capping on a handful of real events repeated many times over, silently
biasing the sample toward whichever encounters happened to be photographed
most frequently.

An event is reconstructed as: images at the same `location`, of the same
`species`, within `burst_window_seconds` of each other. Rows are sorted by
(location, species, datetime); a new event starts whenever the gap to the
previous frame in that (location, species) group exceeds the window.

Rows with no parseable datetime (~7% of the dataset) are never merged into
a burst and never silently dropped — each gets its own singleton event id,
and `has_timestamp=False` is set so downstream steps (seasonal labelling)
can treat them explicitly.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_BURST_WINDOW_SECONDS = 60


def assign_events(
    df: pd.DataFrame,
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
    location_col: str = "location",
    species_col: str = "species",
    datetime_col: str = "datetime",
    id_col: str = "image_id",
) -> pd.DataFrame:
    """Return a copy of `df` with `event_id` and `has_timestamp` columns added."""
    out = df.copy()
    parsed = pd.to_datetime(out[datetime_col], errors="coerce")
    has_ts = parsed.notna()
    out["has_timestamp"] = has_ts

    event_id = pd.Series(index=out.index, dtype=object)

    ts_idx = out.index[has_ts]
    if len(ts_idx):
        sub = out.loc[ts_idx, [location_col, species_col]].copy()
        sub["_dt"] = parsed.loc[ts_idx]
        sub = sub.sort_values([location_col, species_col, "_dt"])

        gap_seconds = sub.groupby([location_col, species_col], sort=False)["_dt"].diff().dt.total_seconds()
        new_event = gap_seconds.isna() | (gap_seconds > burst_window_seconds)
        event_num = new_event.groupby([sub[location_col], sub[species_col]]).cumsum().astype(int)

        sub_event_id = (
            sub[location_col].astype(str)
            + "|" + sub[species_col].astype(str)
            + "|evt" + event_num.astype(str)
        )
        event_id.loc[sub.index] = sub_event_id

    missing_idx = out.index[~has_ts]
    if len(missing_idx):
        event_id.loc[missing_idx] = (
            out.loc[missing_idx, location_col].astype(str)
            + "|" + out.loc[missing_idx, species_col].astype(str)
            + "|singleton|" + out.loc[missing_idx, id_col].astype(str)
        )

    out["event_id"] = event_id
    return out
