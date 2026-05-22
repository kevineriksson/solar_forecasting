"""Clear-sky irradiance via pvlib for the fixed Wolf Point, MT site.

Uses the Ineichen model with pvlib's bundled Linke turbidity climatology — no
external inputs required beyond location and timestamps.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pvlib

CLEARSKY_COLUMNS = ("cs_ghi", "cs_dni", "cs_dhi")


def build_location(site: Mapping[str, object]) -> pvlib.location.Location:
    return pvlib.location.Location(
        latitude=float(site["latitude"]),  # type: ignore[arg-type]
        longitude=float(site["longitude"]),  # type: ignore[arg-type]
        tz=str(site["timezone"]),
        altitude=float(site["altitude_m"]),  # type: ignore[arg-type]
        name=str(site.get("name", "site")),
    )


def compute_clearsky(index: pd.DatetimeIndex, site: Mapping[str, object]) -> pd.DataFrame:
    """Return a DataFrame with cs_ghi, cs_dni, cs_dhi indexed by `index`.

    `index` must be a UTC-localized DatetimeIndex.
    """
    if index.tz is None:
        raise ValueError("clearsky index must be timezone-aware (UTC)")
    loc = build_location(site)
    cs = loc.get_clearsky(index, model="ineichen")
    cs = cs.rename(columns={"ghi": "cs_ghi", "dni": "cs_dni", "dhi": "cs_dhi"})
    return cs[list(CLEARSKY_COLUMNS)].astype("float64")
