"""
Logistics Data Analysis - Week 1 Strategic Plan (proof-of-concept)
Scenario: Regional distributor with 1 warehouse (DC) serving 25 retail stores.
Goals: forecast demand, segment stores, set inventory policy, plan delivery routes.
Run:  python logistics_analysis.py
(Uses synthetic data; replace generate_data() with a real CSV loader, e.g. Kaggle datasets.)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

rng = np.random.default_rng(42)

# ---------- 1. DATA COLLECTION (synthetic stand-in) ----------
def generate_data(n_stores=25, n_days=365):
    stores = pd.DataFrame({
        "store_id": range(1, n_stores + 1),
        "x_km": rng.uniform(-50, 50, n_stores),
        "y_km": rng.uniform(-50, 50, n_stores),
        "base_demand": rng.integers(40, 200, n_stores),
    })
    dates = pd.date_range("2025-01-01", periods=n_days)
    rows = []
    for _, s in stores.iterrows():
        dow = dates.dayofweek.values
        season = 1 + 0.25 * np.sin(2 * np.pi * dates.dayofyear.values / 365)
        weekend = np.where(dow >= 5, 1.3, 1.0)
        promo = rng.binomial(1, 0.1, n_days)
        demand = s.base_demand * season * weekend * (1 + 0.4 * promo) + rng.normal(0, 8, n_days)
        rows.append(pd.DataFrame({"date": dates, "store_id": s.store_id,
                                  "promo": promo, "demand": demand.clip(0).round()}))
    sales = pd.concat(rows, ignore_index=True)
    # delivery log
    sales["lead_time_days"] = rng.normal(2.5, 0.8, len(sales)).clip(1, 7).round(1)
    sales["on_time"] = (rng.random(len(sales)) < 0.88).astype(int)
    sales["in_full"] = (rng.random(len(sales)) < 0.94).astype(int)
    return stores, sales

stores, sales = generate_data()

# ---------- 2. CLEANING ----------
sales = sales.drop_duplicates().dropna()
sales = sales[sales["demand"] >= 0]

# ---------- 3. KPIs ----------
sales["otif"] = sales["on_time"] & sales["in_full"]
kpi = {
    "OTIF %": sales["otif"].mean() * 100,
    "Avg lead time (days)": sales["lead_time_days"].mean(),
    "Total demand (units)": sales["demand"].sum(),
}

# ---------- 4. EXPLORATORY ANALYSIS ----------
daily = sales.groupby("date")["demand"].sum()
fig, ax = plt.subplots(figsize=(9, 3.5))
daily.plot(ax=ax, color="#1f4e79")
ax.set_title("Total Daily Demand Across All Stores"); ax.set_ylabel("Units")
plt.tight_layout(); plt.savefig("outputs/daily_demand.png", dpi=150); plt.close()

# ---------- 5. PREDICTIVE MODEL: demand forecasting ----------
df = sales.copy()
df["dow"] = df["date"].dt.dayofweek
df["doy"] = df["date"].dt.dayofyear
df = df.sort_values(["store_id", "date"])
df["lag7"] = df.groupby("store_id")["demand"].shift(7)
df["roll7"] = df.groupby("store_id")["demand"].transform(lambda s: s.shift(1).rolling(7).mean())
df = df.dropna()
features = ["store_id", "promo", "dow", "doy", "lag7", "roll7"]
cut = df["date"].max() - pd.Timedelta(days=60)          # time-based split
train, test = df[df["date"] <= cut], df[df["date"] > cut]

lin = LinearRegression().fit(train[features], train["demand"])
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(train[features], train["demand"])
results = {}
for name, m in [("Linear Regression", lin), ("Random Forest", rf)]:
    p = m.predict(test[features])
    results[name] = {"MAE": mean_absolute_error(test["demand"], p),
                     "MAPE %": mean_absolute_percentage_error(test["demand"], p) * 100}
print("\nDemand forecast results\n", pd.DataFrame(results).T.round(2))

# ---------- 6. CLUSTERING: store segmentation ----------
profile = sales.groupby("store_id").agg(avg_demand=("demand", "mean"),
                                        demand_std=("demand", "std"),
                                        avg_lead=("lead_time_days", "mean")).reset_index()
profile["cv"] = profile["demand_std"] / profile["avg_demand"]
X = StandardScaler().fit_transform(profile[["avg_demand", "cv", "avg_lead"]])
profile["segment"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
print("\nStore segments\n", profile.groupby("segment")[["avg_demand", "cv"]].mean().round(2))

# ---------- 7. INVENTORY OPTIMIZATION: safety stock & reorder point ----------
Z = 1.65                                    # 95% service level
profile["safety_stock"] = Z * profile["demand_std"] * np.sqrt(profile["avg_lead"])
profile["reorder_point"] = profile["avg_demand"] * profile["avg_lead"] + profile["safety_stock"]
S, H = 50, 2.0                              # order cost, holding cost per unit/yr
profile["EOQ"] = np.sqrt(2 * profile["avg_demand"] * 365 * S / H)

# ---------- 8. ROUTE OPTIMIZATION: nearest neighbour + 2-opt ----------
pts = stores[["x_km", "y_km"]].values
depot = np.array([0.0, 0.0])
coords = np.vstack([depot, pts])
D = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

def route_len(r): return sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))

def nearest_neighbour(n):
    route, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[route[-1], j]); route.append(nxt); left.remove(nxt)
    return route + [0]

def two_opt(route):
    best, improved = route, True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_len(cand) < route_len(best) - 1e-9:
                    best, improved = cand, True
    return best

naive = list(range(len(coords))) + [0]
nn = nearest_neighbour(len(coords)); opt = two_opt(nn)
print(f"\nRoute km  -> unoptimised: {route_len(naive):.0f} | NN: {route_len(nn):.0f} | 2-opt: {route_len(opt):.0f}")
print(f"Distance saved vs unoptimised: {(1 - route_len(opt)/route_len(naive))*100:.1f}%")

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(coords[opt, 0], coords[opt, 1], "-o", color="#1f4e79", ms=4)
ax.plot(*depot, "r*", ms=15, label="Warehouse"); ax.legend(); ax.set_title("Optimised Delivery Route")
plt.tight_layout(); plt.savefig("outputs/route.png", dpi=150); plt.close()

print("\nKPI baseline\n", {k: round(v, 2) for k, v in kpi.items()})
