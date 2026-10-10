"""
Shared design conventions for the Streamlit review dashboard.

The actual visual theme (colors, fonts, radius, chart palette, light/dark
mode) lives entirely in .streamlit/config.toml -- Streamlit's own bundled
theming guidance is explicit that native config.toml theming should be
used over custom CSS/HTML for colors/fonts/backgrounds ("more
maintainable, won't break with Streamlit updates"). config.toml maps the
project's validated dataviz palette (references/palette.md) onto
Streamlit's native schema for both light and dark mode, giving a real
native theme toggle rather than a CSS approximation.

This module holds the one thing native theming genuinely can't do: a
mapping from this project's own status vocabulary (good / warning /
serious / critical -- the dataviz skill's fixed status roles) onto
st.badge's fixed set of named colors, so every view uses the same
status-to-color mapping rather than each guessing its own.
"""
from __future__ import annotations

import streamlit as st

# dataviz skill's status roles -> st.badge's nearest native named color +
# icon. st.badge only accepts a fixed enum of named colors (not arbitrary
# hex), so this is the closest available mapping, not an exact hex match
# -- config.toml's redColor/orangeColor/yellowColor/greenColor entries
# are what pull those named colors to the palette's actual validated
# status hexes.
_STATUS_TO_BADGE = {
    "good": ("green", ":material/check_circle:"),
    "warning": ("yellow", ":material/error:"),
    "serious": ("orange", ":material/warning:"),
    "critical": ("red", ":material/cancel:"),
}


def status_badge(status: str, label: str) -> None:
    """Renders a status badge using this project's fixed status
    vocabulary (never a categorical/series color -- see dataviz skill's
    status-palette rules). `status` must be one of good/warning/serious/
    critical."""
    color, icon = _STATUS_TO_BADGE[status]
    st.badge(label, icon=icon, color=color)
