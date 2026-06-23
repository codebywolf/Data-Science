# End-to-End E-commerce Analytics: Logistics, Business Intelligence, and Merchant Segmentation

## Executive Summary

This project implements a full-stack analytics pipeline, from data engineering to machine learning, using the Olist Brazilian E-commerce dataset. It constructs an optimized database to analyze logistical efficiency, derives key business intelligence insights, and performs unsupervised K-Means clustering to segment merchant behavior. The core objective is to identify the operational drivers of negative customer reviews and to create actionable merchant archetypes using RFM analysis.

## Architecture & Tech Stack

The project is built with a focus on production-grade, modular components.

*   **Database:** MySQL, SQLite
*   **Data Manipulation & Analysis:** Python (Pandas, NumPy)
*   **Machine Learning:** Scikit-Learn
*   **Data Visualization:** Seaborn, Matplotlib
*   **ETL & Database Interface:** SQLAlchemy

### Repository Structure

```
.
├── database/
│   └── olist.db
├── datasets/
│   └── *.csv
├── notebooks/
│   ├── data_exploration_eda.ipynb
│   └── KMeans_Clustering.ipynb
├── sql/
│   ├── 01_data_exploration.sql
│   ├── 02_rfm_scorecard_abt.sql
│   ├── 03_extended_view.sql
│   └── 04_business_queries.sql
└── src/
    └── csv_to_db.py
```

## Database Engineering & Performance Optimization

The foundation of this project is a robust, optimized database designed for analytical workloads.

*   **Automated ETL Pipeline:** The `src/csv_to_db.py` script provides an automated pipeline for migrating raw CSV data into a structured SQLite database using Python and SQLAlchemy. This ensures reproducibility and scalability.
*   **Data Cleaning & Standardization:** Initial data exploration revealed critical anomalies, including irregularities in 2016 beta data. The pipeline standardizes all timezone information from UTC to `America/Sao_Paulo` for accurate logistical analysis.
*   **Indexed Master View:** To optimize query performance, `sql/03_extended_view.sql` compiles multiple JOINs into a single, indexed view (`master_table_extended`). This denormalized table serves as the primary source for all subsequent BI and ML tasks, drastically reducing query times.

## Business Intelligence & Analytics Insights

The BI portion of this project focuses on uncovering actionable insights from operational data. The queries are located in `sql/04_business_queries.sql`.

A key finding is the direct causal link between logistical performance and customer satisfaction. Analysis demonstrates that **shipping delays and shrinking delivery safety buffers (the time between actual delivery and the estimated date) are the strongest predictors of bad review scores (ratings of 1 or 2)**. This insight allows the business to focus on supply chain optimization as a direct lever for improving customer sentiment.

Other tracked metrics include:
*   Top-performing and underperforming seller cohorts.
*   Geographical and seasonal sales trends.
*   Bad review rates correlated with product categories.

## Machine Learning Pipeline (RFM Merchant Segmentation)

To move beyond descriptive analytics, an unsupervised machine learning pipeline was developed to segment merchants into behavioral archetypes.

*   **Feature Engineering:** An RFM (Recency, Frequency, Monetary) matrix was constructed for **Sellers**, not customers. This decision was made because the dataset exhibited low repeat-purchase variance among customers, making seller-side analysis a more valuable source of insight.
*   **Feature Scaling:** The RFM features were scaled using `StandardScaler` from Scikit-Learn to normalize their distributions and prepare them for clustering.
*   **Optimal Cluster Selection:** The Elbow Method (analyzing Within-Cluster Sum of Squares / Inertia) was used to determine the optimal number of clusters (K) for the K-Means algorithm.
*   **Merchant Archetypes:** The resulting clusters represent distinct merchant profiles, such as:
    *   **Core Revenue Whales:** High-frequency, high-monetary value sellers who form the backbone of the platform.
    *   **Churn Risks:** Recently active but low-frequency sellers who may require engagement to retain.
    *   **New Potentials:** New sellers with high initial sales, representing a growth opportunity.

## How to Run

1.  **Clone Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run ETL Pipeline:**
    Execute the Python script to build the SQLite database from the raw datasets.
    ```bash
    python src/csv_to_db.py
    ```
4.  **Execute SQL Scripts:**
    Run the SQL scripts in the `sql/` directory in numerical order (01 to 04) against the `database/olist.db` file.
5.  **Explore Notebooks:**
    Launch Jupyter and navigate to the `notebooks/` directory to step through the EDA and K-Means clustering analysis.
