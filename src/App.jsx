import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import dashboardData from "./data/dashboardData.json";

const palette = {
  ink: "#102033",
  muted: "#65758b",
  blue: "#2563eb",
  cyan: "#06b6d4",
  green: "#16a34a",
  amber: "#f59e0b",
  red: "#ef4444",
  purple: "#7c3aed",
  slate: "#334155",
};

const segmentColors = {
  "At Risk": palette.red,
  Loyal: palette.blue,
  "Needs Attention": palette.amber,
  Champions: palette.green,
  New: palette.cyan,
};

Highcharts.setOptions({
  colors: [palette.blue, palette.cyan, palette.green, palette.amber, palette.purple, palette.red],
  chart: {
    backgroundColor: "transparent",
    style: {
      fontFamily:
        'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
  },
  title: {
    align: "left",
    style: {
      color: palette.ink,
      fontSize: "17px",
      fontWeight: "700",
    },
  },
  credits: {
    enabled: false,
  },
  legend: {
    itemStyle: {
      color: palette.slate,
      fontWeight: "600",
    },
  },
  xAxis: {
    labels: {
      style: {
        color: palette.muted,
      },
    },
  },
  yAxis: {
    title: {
      style: {
        color: palette.muted,
      },
    },
    labels: {
      style: {
        color: palette.muted,
      },
    },
    gridLineColor: "#e2e8f0",
  },
});

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "BRL",
  maximumFractionDigits: 0,
});

const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function formatCurrency(value) {
  return currencyFormatter.format(value);
}

function formatCompact(value) {
  return compactFormatter.format(value);
}

function ChartCard({ children, className = "" }) {
  return <section className={`chart-card ${className}`}>{children}</section>;
}

function KpiCard({ label, value, helper, tone }) {
  return (
    <article className={`kpi-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{helper}</small>
    </article>
  );
}

function App() {
  const {
    generatedAt,
    kpis,
    monthlyRevenue,
    topCategories,
    paymentTypes,
    stateRevenue,
    segments,
    rfmScatter,
    churnWatchlist,
    cohorts,
    orderStatuses,
  } = dashboardData;

  const monthlyRevenueOptions = {
    chart: {
      type: "areaspline",
      height: 360,
    },
    title: {
      text: "Monthly revenue trend",
    },
    subtitle: {
      text: "Payment revenue from Olist orders, grouped by purchase month",
      align: "left",
    },
    xAxis: {
      categories: monthlyRevenue.map((item) => item.month),
      tickInterval: 2,
      crosshair: true,
    },
    yAxis: {
      title: {
        text: "Revenue",
      },
      labels: {
        formatter() {
          return formatCompact(this.value);
        },
      },
    },
    tooltip: {
      pointFormatter() {
        return `<b>${formatCurrency(this.y)}</b>`;
      },
    },
    plotOptions: {
      areaspline: {
        fillOpacity: 0.16,
        marker: {
          radius: 3,
        },
      },
    },
    series: [
      {
        name: "Revenue",
        data: monthlyRevenue.map((item) => item.revenue),
      },
    ],
  };

  const categoryOptions = {
    chart: {
      type: "bar",
      height: 420,
    },
    title: {
      text: "Top product categories",
    },
    subtitle: {
      text: "Ranked by order item volume",
      align: "left",
    },
    xAxis: {
      categories: topCategories.map((item) => item.category),
      title: null,
    },
    yAxis: {
      min: 0,
      title: {
        text: "Order items",
      },
      labels: {
        formatter() {
          return formatCompact(this.value);
        },
      },
    },
    tooltip: {
      pointFormatter() {
        const category = topCategories[this.index];
        return `<b>${formatCompact(this.y)} items</b><br/>${formatCurrency(category.revenue)} item revenue`;
      },
    },
    series: [
      {
        name: "Order items",
        data: topCategories.map((item) => item.orders),
      },
    ],
  };

  const paymentOptions = {
    chart: {
      type: "pie",
      height: 360,
    },
    title: {
      text: "Payment method mix",
    },
    tooltip: {
      pointFormatter() {
        return `<b>${this.percentage.toFixed(1)}%</b><br/>${formatCurrency(this.y)}`;
      },
    },
    plotOptions: {
      pie: {
        innerSize: "62%",
        borderWidth: 0,
        dataLabels: {
          enabled: true,
          format: "{point.name}: {point.percentage:.1f}%",
          style: {
            color: palette.slate,
            textOutline: "none",
          },
        },
      },
    },
    series: [
      {
        name: "Revenue",
        data: paymentTypes.map((item) => ({
          name: item.type,
          y: item.revenue,
        })),
      },
    ],
  };

  const stateOptions = {
    chart: {
      type: "column",
      height: 360,
    },
    title: {
      text: "Revenue by customer state",
    },
    subtitle: {
      text: "Top Brazilian states by order payment revenue",
      align: "left",
    },
    xAxis: {
      categories: stateRevenue.slice(0, 12).map((item) => item.state),
      crosshair: true,
    },
    yAxis: {
      title: {
        text: "Revenue",
      },
      labels: {
        formatter() {
          return formatCompact(this.value);
        },
      },
    },
    tooltip: {
      pointFormatter() {
        const state = stateRevenue[this.index];
        return `<b>${formatCurrency(this.y)}</b><br/>${formatCompact(state.orders)} orders`;
      },
    },
    series: [
      {
        name: "Revenue",
        data: stateRevenue.slice(0, 12).map((item) => item.revenue),
      },
    ],
  };

  const segmentOptions = {
    chart: {
      type: "pie",
      height: 330,
    },
    title: {
      text: "Customer segments",
    },
    tooltip: {
      pointFormatter() {
        return `<b>${formatCompact(this.y)} customers</b><br/>${this.percentage.toFixed(1)}%`;
      },
    },
    plotOptions: {
      pie: {
        innerSize: "58%",
        borderWidth: 0,
        dataLabels: {
          enabled: true,
          format: "{point.name}",
          style: {
            color: palette.slate,
            textOutline: "none",
          },
        },
      },
    },
    series: [
      {
        name: "Customers",
        data: segments.map((item) => ({
          name: item.segment,
          y: item.customers,
          color: segmentColors[item.segment],
        })),
      },
    ],
  };

  const scatterSeries = segments.map(({ segment }) => ({
    name: segment,
    color: segmentColors[segment],
    data: rfmScatter
      .filter((point) => point.segment === segment)
      .map((point) => ({
        x: point.recency,
        y: point.monetary,
        name: point.customer,
        frequency: point.frequency,
        marker: {
          radius: Math.min(12, 4 + point.frequency * 1.8),
        },
      })),
  }));

  const rfmOptions = {
    chart: {
      type: "scatter",
      height: 420,
      zoomType: "xy",
    },
    title: {
      text: "RFM value concentration",
    },
    subtitle: {
      text: "Top 500 customers by monetary value. Marker size reflects purchase frequency.",
      align: "left",
    },
    xAxis: {
      title: {
        text: "Recency in days",
      },
    },
    yAxis: {
      title: {
        text: "Monetary value",
      },
      labels: {
        formatter() {
          return formatCompact(this.value);
        },
      },
    },
    tooltip: {
      pointFormatter() {
        return `<b>${this.name}</b><br/>${formatCurrency(this.y)}<br/>${this.x} days since last order<br/>${this.frequency} orders`;
      },
    },
    series: scatterSeries,
  };

  const statusOptions = {
    chart: {
      type: "bar",
      height: 330,
    },
    title: {
      text: "Order status health",
    },
    xAxis: {
      categories: orderStatuses.map((item) => item.status),
    },
    yAxis: {
      title: {
        text: "Orders",
      },
      labels: {
        formatter() {
          return formatCompact(this.value);
        },
      },
    },
    series: [
      {
        name: "Orders",
        data: orderStatuses.map((item) => item.orders),
      },
    ],
  };

  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">Olist ecommerce intelligence</p>
          <h1>Executive analytics dashboard</h1>
          <p className="hero-copy">
            A React and Highcharts interface for revenue monitoring, product
            mix, customer segmentation, retention, and churn watchlists.
          </p>
        </div>
        <div className="hero-panel">
          <span>Data refresh</span>
          <strong>{new Date(generatedAt).toLocaleString()}</strong>
          <small>Generated from the CSV datasets in this repository.</small>
        </div>
      </section>

      <section className="kpi-grid" aria-label="Dashboard KPIs">
        <KpiCard
          label="Total revenue"
          value={formatCurrency(kpis.revenue)}
          helper={`${formatCompact(kpis.orders)} paid orders`}
          tone="blue"
        />
        <KpiCard
          label="Unique customers"
          value={formatCompact(kpis.customers)}
          helper={`${formatCurrency(kpis.averageOrderValue)} average order value`}
          tone="cyan"
        />
        <KpiCard
          label="Review score"
          value={kpis.averageReviewScore.toFixed(2)}
          helper="Average customer review rating"
          tone="green"
        />
        <KpiCard
          label="At-risk customers"
          value={formatCompact(
            segments.find((item) => item.segment === "At Risk")?.customers || 0,
          )}
          helper="Based on recency over 180 days"
          tone="red"
        />
      </section>

      <section className="dashboard-grid">
        <ChartCard className="span-2">
          <HighchartsReact highcharts={Highcharts} options={monthlyRevenueOptions} />
        </ChartCard>
        <ChartCard>
          <HighchartsReact highcharts={Highcharts} options={paymentOptions} />
        </ChartCard>
        <ChartCard>
          <HighchartsReact highcharts={Highcharts} options={categoryOptions} />
        </ChartCard>
        <ChartCard>
          <HighchartsReact highcharts={Highcharts} options={stateOptions} />
        </ChartCard>
        <ChartCard>
          <HighchartsReact highcharts={Highcharts} options={segmentOptions} />
        </ChartCard>
        <ChartCard className="span-2">
          <HighchartsReact highcharts={Highcharts} options={rfmOptions} />
        </ChartCard>
        <ChartCard>
          <HighchartsReact highcharts={Highcharts} options={statusOptions} />
        </ChartCard>
      </section>

      <section className="insight-grid">
        <article className="panel">
          <div className="panel-heading">
            <p className="eyebrow">Retention</p>
            <h2>Cohort retention snapshot</h2>
          </div>
          <div className="cohort-table" role="table" aria-label="Cohort retention table">
            <div className="cohort-row cohort-head" role="row">
              <span>Cohort</span>
              <span>Size</span>
              {[0, 1, 2, 3, 4, 5].map((month) => (
                <span key={month}>M{month}</span>
              ))}
            </div>
            {cohorts.map((cohort) => (
              <div className="cohort-row" role="row" key={cohort.cohort}>
                <strong>{cohort.cohort}</strong>
                <span>{formatCompact(cohort.size)}</span>
                {cohort.retention.map((value, index) => (
                  <span
                    className="retention-cell"
                    style={{ "--retention": `${Math.max(value, 3)}%` }}
                    key={`${cohort.cohort}-${index}`}
                  >
                    {value.toFixed(1)}%
                  </span>
                ))}
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <p className="eyebrow">Churn</p>
            <h2>High-value at-risk customers</h2>
          </div>
          <div className="watchlist">
            {churnWatchlist.map((customer) => (
              <div className="watchlist-row" key={customer.customer}>
                <div>
                  <strong>{customer.customer.slice(0, 12)}...</strong>
                  <span>{customer.recency} days since last purchase</span>
                </div>
                <div>
                  <strong>{formatCurrency(customer.monetary)}</strong>
                  <span>{customer.frequency} orders</span>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}

export default App;
