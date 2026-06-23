-- 02_rfm_scorecard_abt.sql

-- Step 1: Clean, consolidate, and isolate core delivered transactions
CREATE VIEW view_customer_rfm_scorecard_abt AS
WITH CleanedMasterOrders AS (
	SELECT
		c.customer_unique_id,
        c.customer_id,
        o.order_id,
        o.order_purchase_timestamp,
        op.payment_value,
        ROW_NUMBER() OVER (
			PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp ASC
        ) AS purchase_sequence
	FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_payments op ON o.order_id = op.order_id
    WHERE o.order_status = 'delivered'
),

-- Step 2: Time-gap analysis (Inter-purchase intervals) using LAG()
BehavioralIntervals AS (
	SELECT
		customer_unique_id,
        order_id,
        order_purchase_timestamp,
        payment_value,
        purchase_sequence,
        LAG(order_purchase_timestamp) OVER (
			PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp ASC
        ) AS previous_order_timestamp
    FROM CleanedMasterOrders
),

-- Step 3: Base RFM aggregation per unique customer profile
CoreMetrics AS (
	SELECT
		customer_unique_id,
        -- Recency calculated against the frozen dataset upper bound
        DATEDIFF(
			(SELECT MAX(order_purchase_timestamp) FROM orders),
            MAX(order_purchase_timestamp)
		) + 1 AS recency,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(payment_value) AS monetary,
        -- Operational metrics
        MIN(order_purchase_timestamp) AS first_purchase_date,
        MAX(order_purchase_timestamp) AS last_purchase_date,
        -- Average days between sequential purchases for repeat buyers
        AVG(DATEDIFF(order_purchase_timestamp, previous_order_timestamp)) AS avg_days_between_purchases
	FROM BehavioralIntervals
    GROUP BY customer_unique_id
)

SELECT
	customer_unique_id,
    ROUND(recency, 2) AS recency,
    frequency,
    ROUND(monetary, 2) AS monetary,
    ROUND(avg_days_between_purchases, 2) AS avg_days_between_purchases,
    NTILE(4) OVER (ORDER BY recency ASC) AS r_score,
    NTILE(4) OVER (ORDER BY frequency DESC) AS f_score,
    NTILE(4) OVER (ORDER BY monetary DESC) AS m_score,
    -- Global revenue ranking 
    ROW_NUMBER() OVER (ORDER BY monetary ASC) AS global_revenue_rank
FROM CoreMetrics;

SELECT * FROM view_customer_rfm_scorecard_abt;

SELECT 
    -- 1. Count rows where the metric is NULL (One-time buyers)
    SUM(CASE WHEN avg_days_between_purchases IS NULL THEN 1 ELSE 0 END) AS null_count,
    -- 2. Count total rows in the entire customer table
    COUNT(*) AS total_count,
    -- 3. Perform safe division to get the precise percentage
    ROUND(
        (SUM(CASE WHEN avg_days_between_purchases IS NULL THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)) * 100, 2
    ) AS null_percentage
FROM view_customer_rfm_scorecard_abt;

/*
Insight: The One-Time Buyer Dominance

Our conditional aggregation query reveals that 94.15% of unique customers have only placed a single order within the tracking window. 

- Analytical Nuance: This extreme structural skew creates a massive density spike at Frequency = 1, making feature transformation (Log/Power scaling) essential before executing KMeans clustering.

- Strategic Recommendation: Standard retention campaigns targeting high-recency drops are inefficient here. The business must prioritize automatic, post-purchase email triggers within 7–14 days of delivery to convert first-time buyers into secondary cohorts.
*/