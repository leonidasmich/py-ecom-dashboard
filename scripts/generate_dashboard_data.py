import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT = ROOT / "src" / "data" / "dashboardData.json"


def read_csv(name):
    with (DATA_DIR / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_datetime(value):
    if not value:
        return None
    return datetime.fromisoformat(value)


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def money(value):
    return round(float(value or 0), 2)


def build_dashboard_data():
    customers = {
        row["customer_id"]: {
            "unique_id": row["customer_unique_id"],
            "state": row["customer_state"],
            "city": row["customer_city"],
        }
        for row in read_csv("olist_customers_dataset.csv")
    }

    translations = {}
    for row in read_csv("product_category_name_translation.csv"):
        source = row.get("product_category_name") or row.get("\ufeffproduct_category_name")
        if source:
            translations[source] = row["product_category_name_english"]

    products = {}
    for row in read_csv("olist_products_dataset.csv"):
        category = row.get("product_category_name") or "uncategorized"
        products[row["product_id"]] = translations.get(category, category)

    orders = {}
    for row in read_csv("olist_orders_dataset.csv"):
        purchased_at = parse_datetime(row["order_purchase_timestamp"])
        customer = customers.get(row["customer_id"], {})
        orders[row["order_id"]] = {
            "customer_id": row["customer_id"],
            "customer_unique_id": customer.get("unique_id", row["customer_id"]),
            "state": customer.get("state", "Unknown"),
            "status": row["order_status"],
            "purchased_at": purchased_at,
            "month": purchased_at.strftime("%Y-%m") if purchased_at else "Unknown",
        }

    payment_by_order = defaultdict(float)
    payment_type_counts = Counter()
    payment_type_values = defaultdict(float)
    for row in read_csv("olist_order_payments_dataset.csv"):
        order_id = row["order_id"]
        value = money(row["payment_value"])
        payment_by_order[order_id] += value
        payment_type = row["payment_type"].replace("_", " ").title()
        payment_type_counts[payment_type] += 1
        payment_type_values[payment_type] += value

    review_scores = [
        int(row["review_score"])
        for row in read_csv("olist_order_reviews_dataset.csv")
        if row.get("review_score")
    ]

    monthly_revenue = defaultdict(float)
    state_revenue = defaultdict(float)
    state_orders = Counter()
    customer_orders = defaultdict(set)
    customer_revenue = defaultdict(float)
    customer_last_purchase = {}
    customer_first_month = {}
    customer_months = defaultdict(set)
    status_counts = Counter()

    for order_id, order in orders.items():
        value = payment_by_order.get(order_id, 0)
        if not value or not order["purchased_at"]:
            continue

        customer_id = order["customer_unique_id"]
        monthly_revenue[order["month"]] += value
        state_revenue[order["state"]] += value
        state_orders[order["state"]] += 1
        customer_orders[customer_id].add(order_id)
        customer_revenue[customer_id] += value
        customer_months[customer_id].add(order["month"])
        status_counts[order["status"].replace("_", " ").title()] += 1

        purchased_at = order["purchased_at"]
        if purchased_at > customer_last_purchase.get(customer_id, datetime.min):
            customer_last_purchase[customer_id] = purchased_at
        month = order["month"]
        if month < customer_first_month.get(customer_id, "9999-99"):
            customer_first_month[customer_id] = month

    category_counts = Counter()
    category_revenue = defaultdict(float)
    for row in read_csv("olist_order_items_dataset.csv"):
        category = products.get(row["product_id"], "Uncategorized")
        category_counts[category] += 1
        category_revenue[category] += money(row["price"]) + money(row["freight_value"])

    snapshot = max(customer_last_purchase.values())
    monetary_values = list(customer_revenue.values())
    high_value_threshold = percentile(monetary_values, 0.75)

    segment_counts = Counter()
    rfm_points = []
    churn_candidates = []
    for customer_id, last_purchase in customer_last_purchase.items():
        recency = (snapshot - last_purchase).days
        frequency = len(customer_orders[customer_id])
        monetary = customer_revenue[customer_id]

        if recency <= 60 and frequency >= 2 and monetary >= high_value_threshold:
            segment = "Champions"
        elif frequency >= 2:
            segment = "Loyal"
        elif recency <= 60:
            segment = "New"
        elif recency > 180:
            segment = "At Risk"
        else:
            segment = "Needs Attention"

        segment_counts[segment] += 1
        point = {
            "customer": customer_id,
            "recency": recency,
            "frequency": frequency,
            "monetary": round(monetary, 2),
            "segment": segment,
        }
        rfm_points.append(point)
        if segment == "At Risk":
            churn_candidates.append(point)

    cohort_members = defaultdict(set)
    cohort_retention = defaultdict(lambda: defaultdict(set))
    for customer_id, first_month in customer_first_month.items():
        cohort_members[first_month].add(customer_id)
        first_year, first_month_number = map(int, first_month.split("-"))
        for active_month in customer_months[customer_id]:
            year, month = map(int, active_month.split("-"))
            offset = (year - first_year) * 12 + (month - first_month_number)
            if 0 <= offset <= 5:
                cohort_retention[first_month][offset].add(customer_id)

    cohorts = []
    for cohort in sorted(cohort_members)[-12:]:
        size = len(cohort_members[cohort])
        cohorts.append(
            {
                "cohort": cohort,
                "size": size,
                "retention": [
                    round(len(cohort_retention[cohort][month]) / size * 100, 1)
                    if size
                    else 0
                    for month in range(6)
                ],
            }
        )

    top_categories = [
        {
            "category": category.replace("_", " ").title(),
            "orders": count,
            "revenue": round(category_revenue[category], 2),
        }
        for category, count in category_counts.most_common(10)
    ]

    data = {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "kpis": {
            "revenue": round(sum(payment_by_order.values()), 2),
            "orders": len(payment_by_order),
            "customers": len(customer_orders),
            "averageOrderValue": round(mean(payment_by_order.values()), 2),
            "averageReviewScore": round(mean(review_scores), 2),
        },
        "monthlyRevenue": [
            {"month": month, "revenue": round(value, 2)}
            for month, value in sorted(monthly_revenue.items())
        ],
        "topCategories": top_categories,
        "paymentTypes": [
            {
                "type": payment_type,
                "count": payment_type_counts[payment_type],
                "revenue": round(payment_type_values[payment_type], 2),
            }
            for payment_type, _ in payment_type_counts.most_common()
        ],
        "stateRevenue": [
            {
                "state": state,
                "revenue": round(revenue, 2),
                "orders": state_orders[state],
            }
            for state, revenue in sorted(
                state_revenue.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "segments": [
            {"segment": segment, "customers": count}
            for segment, count in segment_counts.most_common()
        ],
        "rfmScatter": sorted(rfm_points, key=lambda item: item["monetary"], reverse=True)[:500],
        "churnWatchlist": sorted(
            churn_candidates, key=lambda item: item["monetary"], reverse=True
        )[:8],
        "cohorts": cohorts,
        "orderStatuses": [
            {"status": status, "orders": count}
            for status, count in status_counts.most_common()
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build_dashboard_data()
