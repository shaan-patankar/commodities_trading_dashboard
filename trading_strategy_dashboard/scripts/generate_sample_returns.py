"""Generate sample daily-returns CSVs for VaR scaling.

For each strategy in ``STRATEGY_FILES`` this reads the PnL CSV to obtain the
exact date index and product column names, then writes a matching
``<name>_returns.csv`` containing synthetic but realistic daily *fractional*
returns for each underlying (Student-t innovations with mild fat tails and a
small drift, per-product volatility).

These are placeholders so the VaR feature can be exercised end to end; replace
them with your real underlying-return series (same filenames + column headers).

Run:  python scripts/generate_sample_returns.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root or the scripts/ dir.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.config import RETURNS_FILES, STRATEGY_FILES  # noqa: E402
from dashboard.data import read_strategy_csv  # noqa: E402

# Per-product annualised-ish daily vol seed (fraction). Spreads get lower vol.
_BASE_DAILY_VOL = 0.012
_RNG = np.random.default_rng(20240601)


def _product_vol(product: str, index: int) -> float:
    """Pick a plausible daily vol for a product (spreads quieter than outrights)."""
    vol = _BASE_DAILY_VOL * (0.7 + 0.25 * (index % 4))
    if "/" in product or "Strangle" in product or "Crack" in product:
        vol *= 0.8  # spreads / structures tend to be lower-vol
    return float(vol)


def generate() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("gen-returns")

    for name, pnl_path in STRATEGY_FILES.items():
        try:
            df = read_strategy_csv(pnl_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s (cannot read PnL: %s)", name, exc)
            continue

        products = [c for c in df.columns if c != "date"]
        n = len(df)
        out = {"Date": df["date"].dt.strftime("%Y-%m-%d").to_numpy()}

        for i, product in enumerate(products):
            vol = _product_vol(product, i)
            # Student-t (df=5) innovations -> mild fat tails; tiny positive drift.
            innov = _RNG.standard_t(df=5, size=n) / np.sqrt(5 / 3)  # unit-ish variance
            returns = 0.0001 + vol * innov
            out[product] = np.round(returns, 6)

        returns_path = RETURNS_FILES[name]
        # Write with a leading Date column matching the PnL date format.
        header = ",".join(["Date", *products])
        lines = [header]
        cols = [out[c] for c in ["Date", *products]]
        for row in zip(*cols):
            lines.append(",".join(str(v) for v in row))
        returns_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote %s (%d rows, %d products)", returns_path.name, n, len(products))


if __name__ == "__main__":
    generate()
