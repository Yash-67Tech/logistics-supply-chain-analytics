# Logistics Supply Chain Analytics – Strategic Planning & Proof of Concept

**Author:** Yash Zujam  |  **Role:** Logistics Data Analyst Intern  |  **Task:** Week 1 – Strategic Planning and Data Exploration

## Scenario
A regional distributor operates one warehouse serving 25 retail stores. Management wants to reduce stockouts,
lower delivery cost and improve on-time delivery. This project plans and prototypes a Python analytics workflow.

## KPIs
| KPI | Why it matters |
|---|---|
| On-Time In-Full (OTIF %) | Core service-level measure |
| Forecast accuracy (MAPE %) | Drives inventory decisions |
| Cost / distance per delivery route | Transport efficiency |
| Inventory turnover & stockout rate | Working capital vs. availability |

## Approach
1. Data collection → 2. Cleaning → 3. EDA & KPI baseline → 4. Demand forecasting (Linear Regression, Random Forest)
5. Store clustering (K-Means) → 6. Safety stock / reorder point / EOQ → 7. Route optimisation (Nearest Neighbour + 2-opt)

## Run
```bash
pip install -r requirements.txt
python logistics_analysis.py
```
Charts are saved to `outputs/`. The prototype uses synthetic data; swap `generate_data()` for a real dataset
(e.g. Kaggle *DataCo Smart Supply Chain*, *Olist E-commerce*, *M5 Forecasting*).

## Sample results (synthetic data)
- Random Forest MAPE ≈ 9.1 % vs Linear Regression ≈ 12.3 %
- 2-opt route ≈ 66 % shorter than an unoptimised visiting order
