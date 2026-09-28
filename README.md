# 📊 Customer Segmentation & Lifetime Value Analysis — Superstore

> Segmentasi pelanggan dengan **RFM** dan proyeksi **Customer Lifetime Value (CLV)** menggunakan Python, untuk menentukan siapa yang harus dipertahankan, dikembangkan, dan dimenangkan kembali.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab&logoColor=white)

---

## 📌 Ringkasan Proyek

| | |
|---|---|
| **Dataset** | Superstore (transaksi retail AS, 3 Jan 2014 – 30 Des 2017) |
| **Ukuran data** | 9.994 baris × 21 kolom, tanpa missing value & duplikat |
| **Cakupan** | 793 customer · 5.009 order · 1.862 produk |
| **Tools** | Python, Pandas, NumPy, Matplotlib, Seaborn, Google Colab |
| **Teknik** | Data Profiling, EDA, RFM Segmentation (7 segmen), CLV Projection & Validation, RFM × CLV Cross-Analysis |

## 🎯 Latar Belakang & Tujuan

Tidak semua pelanggan memberi nilai yang sama bagi bisnis. Proyek ini menjawab:

1. **Siapa** pelanggan terbaik berdasarkan perilaku belanja (Recency, Frequency, Monetary)?
2. **Berapa** profit yang diproyeksikan dari tiap pelanggan pada tahun berikutnya?
3. **Strategi** apa yang tepat untuk tiap kelompok pelanggan?

---

## 🔍 Data Understanding

- Kolom tanggal (`order_date`, `ship_date`) dikonversi ke `datetime`.
- Kualitas data: **0 missing value** dan **0 duplikat**.
- Setiap customer hanya punya **1 segmen** (Consumer / Corporate / Home Office), tetapi **766 dari 793 customer (96,6%)** pernah bertransaksi di lebih dari satu region. Karena itu, region tiap customer ditentukan dengan **modus** (region yang paling sering muncul).

## 📈 Exploratory Data Analysis (EDA)

### 1. Tren Penjualan Bulanan
- Terdapat **pola musiman** yang konsisten: penjualan memuncak di akhir tahun (September–Desember) dan turun tajam di awal tahun.
- Puncak tahunan: Sep 2014 ($81,8K) → Nov 2015 ($76,0K) → Des 2016 ($97,0K) → **Nov 2017 ($118,4K)**.
- Titik terendah selalu di Januari–Februari (mis. Feb 2014: $4,5K).

![Tren Sales Bulanan](images/monthly_sales_trend.png)

### 2. Sales & Profit per Kategori

| Kategori | Sales | Profit | Catatan |
|---|---:|---:|---|
| Technology | $836.154 | $145.455 | Sales & profit tertinggi |
| Furniture | ~$742.000 | $18.451 | Sales besar tetapi margin sangat tipis (*profit leakage*) |
| Office Supplies | $719.047 | $122.491 | Margin sehat dan stabil |

![Sales & Profit per Kategori](images/sales_profit_category.png)

### 3. Distribusi Customer per Segmen
**Consumer** mendominasi (409 customer, >50%), diikuti **Corporate** (236) dan **Home Office** (148).

![Customer per Segment](images/customer_per_segment.png)

### 4. Diskon vs Profit
Transaksi tanpa diskon hingga diskon rendah cenderung menghasilkan profit positif. Pada diskon **≥ 30%** profit mulai negatif dan pada diskon **70–80%** kerugian bisa mencapai sekitar **-$6.600** per transaksi. Pemberian diskon besar perlu dikendalikan.

![Discount vs Profit](images/discount_vs_profit.png)

---

## 🧮 Metodologi

### A. RFM Segmentation
Data transaksi diagregasi ke level **order → customer**, lalu dihitung:

- **Recency**: jarak hari sejak pembelian terakhir (snapshot = tanggal terakhir + 1 hari)
- **Frequency**: jumlah order unik
- **Monetary**: total sales

Ketiganya diberi skor 1–5 (quintile) dengan `pd.qcut`, lalu dipetakan ke **7 segmen** berdasarkan kombinasi R dan F:

`Champions` · `Loyal Customers` · `Potential Loyalist` · `New Customers` · `Need Attention` · `At Risk` · `Lost`

### B. Proyeksi CLV (Calibration–Holdout)

```
Calibration : Tahun 1–3 (2014–2016)  → membangun model
Holdout     : Tahun 4   (2017)       → validasi
Prediksi    : Tahun 5               → proyeksi resmi
```

```
AOV                   = total_sales / total_orders
annual_revenue        = AOV × (total_orders / jumlah_tahun)
margin_pct            = total_profit / total_sales
projected_margin_year = annual_revenue × margin_pct
```

Proyeksi customer lama dikombinasikan dengan **estimasi profit customer baru** (rata-rata profit tahun pertama akuisisi per tahun) agar proyeksi lebih realistis.

### C. Segmentasi CLV
Customer dibagi ke **5 tingkat** berdasarkan proyeksi margin tahun ke-5 (~158 customer per tingkat):

`Premium` · `High Value` · `Mid Value` · `Low Value` · `Minimal Value`

---

## 💻 Kode

<details>
<summary><b>1️⃣ Setup, Load Data & Data Understanding</b></summary>

```python
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 999
pd.options.display.float_format = "{:.2f}".format

# Folder untuk menyimpan gambar
os.makedirs("images", exist_ok=True)

def save_fig(name):
    plt.savefig(f"images/{name}.png", dpi=150, bbox_inches="tight")

# Load data (di Google Colab: mount Drive lalu ganti DATA_PATH)
DATA_PATH = "data/superstore_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Konversi tanggal
df["order_date"] = pd.to_datetime(df["order_date"])
df["ship_date"] = pd.to_datetime(df["ship_date"])

# Cek kualitas data
df.info()
print("Missing value:\n", df.isnull().sum())
print("Duplikat :", df.duplicated().sum())
print("Shape    :", df.shape)

print("Jumlah customer unik :", df["customer_id"].nunique())
print("Jumlah order unik    :", df["order_id"].nunique())
print("Jumlah produk unik   :", df["product_id"].nunique())
print("Rentang tanggal      :", df["order_date"].min(), "-", df["order_date"].max())

# Konsistensi region & segment per customer
region_check = df.groupby("customer_id")["region"].nunique()
print("Customer dengan region > 1 :", (region_check > 1).sum(), "dari", df["customer_id"].nunique())
segment_check = df.groupby("customer_id")["segment"].nunique()
print("Customer dengan segment > 1:", (segment_check > 1).sum())
```
</details>

<details>
<summary><b>2️⃣ Exploratory Data Analysis</b></summary>

```python
# Tren sales bulanan
monthly_sales = df.set_index("order_date").resample("ME")["sales"].sum()
monthly_sales.plot(figsize=(12, 4), title="Tren Sales Bulanan")
plt.ylabel("Total Sales ($)")
save_fig("monthly_sales_trend")
plt.show()

# Bulan puncak & terendah tiap tahun
monthly_sales_df = monthly_sales.reset_index()
monthly_sales_df["Tahun"] = monthly_sales_df["order_date"].dt.year
monthly_sales_df["Bulan"] = monthly_sales_df["order_date"].dt.month_name()

idx_max = monthly_sales_df.groupby("Tahun")["sales"].idxmax()
print(monthly_sales_df.loc[idx_max, ["Tahun", "Bulan", "sales"]].to_string(index=False))
idx_min = monthly_sales_df.groupby("Tahun")["sales"].idxmin()
print(monthly_sales_df.loc[idx_min, ["Tahun", "Bulan", "sales"]].to_string(index=False))

# Sales & profit per kategori
cat_summary = df.groupby("category")[["sales", "profit"]].sum().sort_values("sales", ascending=False)
ax = cat_summary.plot(kind="bar", figsize=(9, 6), title="Sales & Profit per Kategori")
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=9)
plt.xticks(rotation=25)
save_fig("sales_profit_category")
plt.show()

# Distribusi customer per segment
plt.figure(figsize=(9, 7))
ax = sns.countplot(data=df.drop_duplicates("customer_id"), x="segment")
ax.bar_label(ax.containers[0], padding=3, fontsize=10)
plt.title("Jumlah Customer per Segment")
ax.set_xlabel("Segment")
ax.set_ylabel("Jumlah Customer")
save_fig("customer_per_segment")
plt.show()

# Diskon vs profit
sns.scatterplot(data=df, x="discount", y="profit", alpha=0.3)
plt.title("Discount vs Profit")
save_fig("discount_vs_profit")
plt.show()
```
</details>

<details>
<summary><b>3️⃣ RFM Segmentation</b></summary>

```python
# Agregasi order level -> customer level
order_level = (
    df.groupby(["order_id", "customer_id", "customer_name", "segment", "region", "order_date"])
    .agg(order_sales=("sales", "sum"), order_profit=("profit", "sum"), order_qty=("quantity", "sum"))
    .reset_index()
)

# Region customer = modus
region_mode = (
    order_level.groupby("customer_id")["region"]
    .agg(lambda x: x.mode().iloc[0])
    .rename("most_common_region")
)

snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

customer_level = (
    order_level.groupby(["customer_id", "customer_name", "segment"])
    .agg(
        last_purchase=("order_date", "max"),
        first_purchase=("order_date", "min"),
        frequency=("order_id", "nunique"),
        monetary=("order_sales", "sum"),
        total_profit=("order_profit", "sum"),
    )
    .reset_index()
)

customer_level = customer_level.merge(region_mode, on="customer_id")
customer_level["recency_days"] = (snapshot_date - customer_level["last_purchase"]).dt.days
customer_level["tenure_days"] = (customer_level["last_purchase"] - customer_level["first_purchase"]).dt.days
customer_level["avg_order_value"] = customer_level["monetary"] / customer_level["frequency"]

# Skor RFM 1-5 (quintile)
customer_level["R_score"] = pd.qcut(customer_level["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
customer_level["F_score"] = pd.qcut(customer_level["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
customer_level["M_score"] = pd.qcut(customer_level["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)


def segment_rfm_7(row):
    """Mapping ke 7 segmen berdasarkan R & F (urutan kondisi penting)."""
    r, f = row["R_score"], row["F_score"]
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 4 and f == 1:
        return "New Customers"
    elif r == 3 and f <= 2:
        return "Need Attention"
    elif r >= 2 and f >= 3:
        return "Loyal Customers"
    elif r >= 3 and f <= 3:
        return "Potential Loyalist"
    elif r <= 2 and f >= 2:
        return "At Risk"
    else:
        return "Lost"


customer_level["rfm_segment"] = customer_level.apply(segment_rfm_7, axis=1)

segment_order = {
    "Champions": 1, "Loyal Customers": 2, "Potential Loyalist": 3,
    "New Customers": 4, "Need Attention": 5, "At Risk": 6, "Lost": 7,
}
customer_level["segment_order"] = customer_level["rfm_segment"].map(segment_order)

# Distribusi segmen
seg_dist = (
    customer_level.groupby(["segment_order", "rfm_segment"])
    .size().reset_index(name="jumlah_customer")
    .sort_values("segment_order")
)
print(seg_dist)

plt.figure(figsize=(9, 5))
ax = sns.barplot(data=seg_dist, x="rfm_segment", y="jumlah_customer", order=seg_dist["rfm_segment"])
ax.bar_label(ax.containers[0], padding=3, fontsize=10)
plt.title("Distribusi Customer per RFM Segment")
plt.xticks(rotation=30)
save_fig("rfm_distribution")
plt.show()
```
</details>

<details>
<summary><b>4️⃣ CLV: Validasi Model (Calibration–Holdout)</b></summary>

```python
# Split: Tahun 1-3 = calibration, Tahun 4 = holdout
min_date = df["order_date"].min()
cutoff_date = min_date + pd.DateOffset(years=3)
df_cal = df[df["order_date"] <= cutoff_date]
df_hold = df[df["order_date"] > cutoff_date]

# Proyeksi profit tahun 4 dari data calibration
cal_customers = (
    df_cal.groupby(["customer_id", "customer_name"])
    .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), total_orders=("order_id", "nunique"))
    .reset_index()
)
cal_customers["AOV"] = cal_customers["total_sales"] / cal_customers["total_orders"]
cal_customers["orders_per_yr"] = cal_customers["total_orders"] / 3.0
cal_customers["annual_revenue"] = cal_customers["AOV"] * cal_customers["orders_per_yr"]
cal_customers["margin_pct"] = cal_customers["total_profit"] / cal_customers["total_sales"]
cal_customers["projected_margin_yr4"] = cal_customers["annual_revenue"] * cal_customers["margin_pct"]

# Bandingkan dengan profit aktual tahun 4
actual_yr4 = df_hold.groupby("customer_id")["profit"].sum().reset_index(name="actual_profit_yr4")
eval_df = pd.merge(cal_customers, actual_yr4, on="customer_id", how="left").fillna({"actual_profit_yr4": 0})
eval_df["abs_error"] = (eval_df["projected_margin_yr4"] - eval_df["actual_profit_yr4"]).abs()

mae = eval_df["abs_error"].mean()
print(f"MAE per Customer           : ${mae:,.2f}")
print(f"Total Proyeksi Profit Thn 4: ${eval_df['projected_margin_yr4'].sum():,.2f}")
print(f"Total Profit Aktual Thn 4  : ${eval_df['actual_profit_yr4'].sum():,.2f}")

# Koreksi: tambahkan estimasi profit customer baru
first_order = df_cal.groupby("customer_id")["order_date"].min().reset_index(name="first_order_date")
first_order["first_year"] = first_order["first_order_date"].dt.year

df_cal_merged = df_cal.merge(first_order[["customer_id", "first_year"]], on="customer_id")
df_cal_merged["order_year"] = df_cal_merged["order_date"].dt.year

new_cust_first_year = df_cal_merged[df_cal_merged["order_year"] == df_cal_merged["first_year"]]
new_cust_profit_avg = new_cust_first_year.groupby("first_year")["profit"].sum().mean()

existing_proj = eval_df["projected_margin_yr4"].sum()
total_projected_yr4 = existing_proj + new_cust_profit_avg
actual_total = eval_df["actual_profit_yr4"].sum()

print(f"Proyeksi Existing Customer : ${existing_proj:,.2f}")
print(f"Estimasi New Customer      : ${new_cust_profit_avg:,.2f}")
print(f"Total Proyeksi Disesuaikan : ${total_projected_yr4:,.2f}")
print(f"Total Profit Aktual Thn 4  : ${actual_total:,.2f}")
print(f"Selisih terhadap aktual    : {abs(total_projected_yr4 - actual_total) / actual_total:.1%}")
```
</details>

<details>
<summary><b>5️⃣ CLV: Proyeksi Tahun 5, Segmentasi & Cross-Analysis</b></summary>

```python
# Proyeksi tahun 5 (memakai seluruh data tahun 1-4)
full_customers = (
    df.groupby(["customer_id", "customer_name"])
    .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"), total_orders=("order_id", "nunique"))
    .reset_index()
)
full_customers["AOV"] = full_customers["total_sales"] / full_customers["total_orders"]
full_customers["orders_per_yr"] = full_customers["total_orders"] / 4.0
full_customers["annual_revenue"] = full_customers["AOV"] * full_customers["orders_per_yr"]
full_customers["margin_pct"] = full_customers["total_profit"] / full_customers["total_sales"]
full_customers["projected_margin_yr5"] = full_customers["annual_revenue"] * full_customers["margin_pct"]

existing_proj_yr5 = full_customers["projected_margin_yr5"].sum()
total_projected_yr5 = existing_proj_yr5 + new_cust_profit_avg
print(f"Proyeksi Existing Customer : ${existing_proj_yr5:,.2f}")
print(f"Estimasi New Customer      : ${new_cust_profit_avg:,.2f}")
print(f"TOTAL PROYEKSI PROFIT THN 5: ${total_projected_yr5:,.2f}")

# Segmentasi CLV (5 tingkat)
clv_segment_map = {5: "Premium", 4: "High Value", 3: "Mid Value", 2: "Low Value", 1: "Minimal Value"}

full_customers["clv_score"] = pd.qcut(
    full_customers["projected_margin_yr5"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
).astype(int)
full_customers["clv_segment"] = full_customers["clv_score"].map(clv_segment_map)
print(full_customers["clv_segment"].value_counts().reindex(clv_segment_map.values()))

# Visualisasi: jumlah customer vs proyeksi margin per segmen CLV
segment_summary = (
    full_customers.groupby("clv_segment", observed=False)
    .agg(total_customers=("customer_id", "count"), total_projected_profit=("projected_margin_yr5", "sum"))
    .reindex(["Minimal Value", "Low Value", "Mid Value", "High Value", "Premium"])
)

fig, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x=segment_summary.index, y=segment_summary["total_customers"],
            hue=segment_summary.index, ax=ax1, palette="Blues_d", legend=False)
ax1.set_ylabel("Jumlah Customer", color="navy", fontsize=12)
ax1.set_xlabel("Segmen CLV", fontsize=12)

ax2 = ax1.twinx()
sns.lineplot(x=segment_summary.index, y=segment_summary["total_projected_profit"],
             ax=ax2, color="red", marker="o", linewidth=2.5)
ax2.set_ylabel("Total Projected Margin Thn 5 ($)", color="red", fontsize=12)
plt.title("Perbandingan Jumlah Customer vs Proyeksi Margin per Segmen", fontsize=14, fontweight="bold")
save_fig("clv_segment")
plt.show()

# Cross-analysis RFM x CLV
rfm_order = ["Champions", "Loyal Customers", "Potential Loyalist",
             "New Customers", "Need Attention", "At Risk", "Lost"]
clv_order = ["Premium", "High Value", "Mid Value", "Low Value", "Minimal Value"]

full_customers = full_customers.merge(
    customer_level[["customer_id", "rfm_segment"]], on="customer_id", how="left"
)
cross = pd.crosstab(full_customers["rfm_segment"], full_customers["clv_segment"])
cross = cross.reindex(index=rfm_order, columns=clv_order).fillna(0).astype(int)
print(cross)

plt.figure(figsize=(10, 6))
sns.heatmap(cross, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Cross-tab: RFM Segment vs CLV Segment")
plt.ylabel("RFM Segment")
plt.xlabel("CLV Segment")
save_fig("rfm_clv_heatmap")
plt.show()
```
</details>

---

## ✅ Hasil

### Distribusi Segmen RFM

| Segmen RFM | Jumlah Customer |
|---|---:|
| Champions | 167 |
| Loyal Customers | 263 |
| Potential Loyalist | 50 |
| New Customers | 39 |
| Need Attention | 57 |
| At Risk | 120 |
| Lost | 97 |

**Champions + Loyal Customers = 430 customer (54,2%)** dari total basis pelanggan.

![Distribusi RFM](images/rfm_distribution.png)

### Validasi Model CLV (Holdout Tahun 4)

| Metrik | Nilai |
|---|---:|
| Proyeksi existing customer | $64.424 |
| Estimasi new customer | $23.905 |
| **Total proyeksi** | **$88.329** |
| **Profit aktual** | **$91.317** |
| Selisih | ≈ 3,3% |
| MAE per customer | $266,10 |

Model awal (hanya existing customer) meng-*underestimate* profit aktual ($64.424 vs $91.317). Setelah komponen customer baru ditambahkan, proyeksi mendekati profit aktual.

### Proyeksi Profit Tahun ke-5

| Komponen | Nilai |
|---|---:|
| Existing customer | $71.599 |
| New customer | $23.905 |
| **Total proyeksi** | **$95.504** |

![Segmen CLV](images/clv_segment.png)

### Cross-Analysis: RFM × CLV

| RFM \ CLV | Premium | High | Mid | Low | Minimal |
|---|---:|---:|---:|---:|---:|
| Champions | 52 | 40 | 35 | 18 | 22 |
| Loyal Customers | 61 | 66 | 48 | 40 | 48 |
| Potential Loyalist | 6 | 9 | 12 | 13 | 10 |
| New Customers | 5 | 3 | 10 | 14 | 7 |
| Need Attention | 7 | 9 | 10 | 19 | 12 |
| At Risk | 23 | 22 | 25 | 23 | 27 |
| Lost | 5 | 9 | 19 | 31 | 33 |

![Heatmap RFM x CLV](images/rfm_clv_heatmap.png)

---

## 💡 Insight & Rekomendasi Bisnis

**Insight utama**

1. **Nilai ekonomi terkonsentrasi di Champions & Loyal.** Dari 159 customer Premium, 113 (±71%) berasal dari dua segmen ini.
2. **Ada nilai besar yang berisiko hilang.** 23 customer berkategori **Premium** justru berada di segmen *At Risk*. Secara total, 45 customer At Risk masuk High Value/Premium.
3. **Frekuensi ≠ nilai.** Sebagian Champions punya CLV rendah (mis. 22 di *Minimal Value*), sehingga RFM saja tidak cukup dan perlu dilengkapi CLV.
4. **Segmen Lost didominasi CLV rendah** (64 dari 97 di Low/Minimal), sehingga tidak perlu investasi besar untuk win-back.

**Rekomendasi prioritas**

| Prioritas | Segmen | Aksi |
|---|---|---|
| 🔴 1 | At Risk + CLV tinggi | Win-back personal (penawaran eksklusif, outreach langsung) |
| 🟢 2 | Champions & Loyal | Program loyalitas, akses awal produk, referral, cross-sell/bundling |
| 🟡 3 | Potential Loyalist & New Customers | Onboarding, dorong repeat order kedua |
| 🟠 4 | Need Attention | Reminder & promo terbatas waktu |
| ⚪ 5 | Lost + CLV rendah | Kampanye otomatis berbiaya rendah |
| ⚠️ Umum | Diskon | Batasi diskon di atas 30% karena menggerus profit |

---

## 📁 Struktur Repository

```
├── README.md
├── notebooks/
│   └── Naela_Customer_Lifetime_Value.ipynb
├── data/
│   └── superstore_dataset.csv
├── images/
│   ├── monthly_sales_trend.png
│   ├── sales_profit_category.png
│   ├── customer_per_segment.png
│   ├── discount_vs_profit.png
│   ├── rfm_distribution.png
│   ├── clv_segment.png
│   └── rfm_clv_heatmap.png
└── report/
    └── Laporan_Customer_Lifetime_Value.pdf
```

## 🚀 Cara Menjalankan

```bash
git clone https://github.com/<username>/<nama-repo>.git
cd <nama-repo>
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook notebooks/Naela_Customer_Lifetime_Value.ipynb
```

Atau buka langsung di Google Colab: **[Buka Notebook](LINK_COLAB_KAMU)**
📄 Laporan lengkap: **[Baca Laporan](LINK_LAPORAN_KAMU)**

## ⚠️ Keterbatasan

- Proyeksi CLV memakai **rata-rata historis sederhana** (AOV × frekuensi × margin) dan bukan model probabilistik seperti BG/NBD + Gamma-Gamma.
- Model mengasumsikan perilaku customer stabil, tanpa memperhitungkan churn probability.
- Rentang data hanya 4 tahun sehingga validasi hanya memakai satu tahun holdout.

**Pengembangan selanjutnya:** BG/NBD & Gamma-Gamma (`lifetimes`), dashboard interaktif Power BI/Looker Studio, K-Means sebagai pembanding RFM.

## 👩‍💻 Author

**Naela**
Data Analyst (Dibimbing Data Analyst & BI Bootcamp) · Accounting graduate
📍 Semarang, Indonesia
🔗 [GitHub](https://github.com/naelasproject0817) · [TikTok / Instagram: Journaela](#) · [LinkedIn](#)

---
*Proyek ini dibuat sebagai bagian dari portofolio data analytics.*
