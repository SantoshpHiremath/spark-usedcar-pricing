"""
Generates synthetic multi-market used-vehicle listing data (international
retail/wholesale pricing across European markets), with realistic pricing-relevant attributes (model,
age, mileage, condition, equipment level) and deliberately injected
data-quality problems, since a used-car pricing analytics pipeline in
practice has to handle messy source data, not a clean textbook table.
"""
from __future__ import annotations

import random

MARKETS = ["DE", "FR", "IT", "ES", "NL"]
MODELS = ["Series 1", "Series 3", "Series 5", "X1", "X3", "X5"]
EQUIPMENT_LEVELS = ["Base", "Advantage", "Sport", "M Sport", "Luxury"]

# Each market has a slightly different price level (currency/market
# conditions) and a different typical listing volume -- a real-world
# characteristic of pricing across European markets that a pipeline
# needs to handle explicitly, not average away.
MARKET_PRICE_MULTIPLIER = {"DE": 1.00, "FR": 1.05, "IT": 0.95, "ES": 0.92, "NL": 1.10}
MARKET_LISTING_COUNTS = {"DE": 900, "FR": 650, "IT": 500, "ES": 420, "NL": 380}

# Base price (EUR) and typical depreciation-relevant attributes per model
MODEL_BASE_PRICE = {
    "Series 1": 24000, "Series 3": 32000, "Series 5": 45000,
    "X1": 33000, "X3": 42000, "X5": 62000,
}
EQUIPMENT_PREMIUM = {"Base": 0, "Advantage": 1500, "Sport": 3500, "M Sport": 6000, "Luxury": 8000}


def _true_price(model: str, age_years: float, mileage_km: int, equipment: str, market: str, rng: random.Random) -> float:
    base = MODEL_BASE_PRICE[model] + EQUIPMENT_PREMIUM[equipment]
    # Depreciation: roughly 15% first year, then ~9%/year after, plus a
    # mileage penalty -- simplified but directionally realistic, not an
    # invented arbitrary formula.
    age_factor = 0.85 * (0.91 ** max(age_years - 1, 0))
    mileage_penalty = max(0, (mileage_km - 15000 * age_years)) * 0.03
    price = base * age_factor - mileage_penalty
    price *= MARKET_PRICE_MULTIPLIER[market]
    price *= rng.uniform(0.95, 1.05)  # market noise
    return max(price, 3000)


def generate_listings(seed: int = 42) -> list:
    rng = random.Random(seed)
    rows = []
    listing_id = 0

    for market in MARKETS:
        n = MARKET_LISTING_COUNTS[market]
        for _ in range(n):
            listing_id += 1
            model = rng.choice(MODELS)
            age_years = round(rng.uniform(0.5, 8.0), 1)
            mileage_km = int(max(1000, rng.gauss(15000 * age_years, 8000)))
            equipment = rng.choices(
                EQUIPMENT_LEVELS, weights=[0.15, 0.30, 0.25, 0.20, 0.10],
            )[0]
            price = round(_true_price(model, age_years, mileage_km, equipment, market, rng), 2)

            row = {
                "listing_id": f"L{listing_id:06d}",
                "market": market,
                "model": model,
                "age_years": age_years,
                "mileage_km": mileage_km,
                "equipment_level": equipment,
                "list_price_eur": price,
                "channel": rng.choices(["retail", "wholesale"], weights=[0.7, 0.3])[0],
            }

            # --- deliberately injected data-quality problems ---
            issue = rng.random()
            if issue < 0.02:
                # duplicate listing (same listing_id re-sent, e.g. from a
                # retry in an upstream feed) -- appended again below
                pass
            elif issue < 0.04:
                row["mileage_km"] = None  # missing mileage
            elif issue < 0.05:
                row["list_price_eur"] = -abs(row["list_price_eur"])  # data-entry sign error
            elif issue < 0.06:
                row["market"] = row["market"].lower()  # inconsistent casing from a source feed

            rows.append(row)
            if issue < 0.02:
                # true duplicate row, identical listing_id
                rows.append(dict(row))

    return rows


def write_csv(path: str, seed: int = 42) -> int:
    import csv
    rows = generate_listings(seed=seed)
    fieldnames = ["listing_id", "market", "model", "age_years", "mileage_km", "equipment_level", "list_price_eur", "channel"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    count = write_csv("data/listings_raw.csv")
    print(f"Generated {count} listing rows across {len(MARKETS)} markets.")
