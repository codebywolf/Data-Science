-- 04_business_queries.sql
-- Purpose: 10 structured queries extracting KPIs from master_table_extended

SELECT * FROM master_table_extended LIMIT 5;

-- Query 1: Monthly Gross Revenue Growth Trend
-- Business Value: Tracks the global sales trajectory and seasonal velocity.
SELECT
	DATE_FORMAT(order_purchase_timestamp, '%Y-%m') AS sales_month,
    ROUND(SUM(price), 2) AS net_revenue,
    ROUND(SUM(gross_item_value), 2) AS gross_revenue,
    COUNT(DISTINCT order_id) AS total_orders
FROM master_table_extended
-- Filtered out 2016 records because Olist was in a fragmented beta-testing phase.
-- True platform launch scaled up in Jan 2017; removing 2016 prevents skewed trends.
WHERE order_purchase_timestamp >= '2017-01-01 00:00:00'
GROUP BY sales_month
ORDER BY sales_month ASC;

-- Query 2: Top 10 Product Categories by Gross Revenue
-- Business Value: Identifies the primary drivers of sales volume and cash flow.
SELECT
	product_category,
	COUNT(DISTINCT order_id) AS total_orders,
    ROUND(SUM(price), 2) AS net_revenue,
    ROUND(SUM(gross_item_value), 2) AS gross_revenue,
    ROUND(AVG(price), 2) AS avg_item_price
FROM master_table_extended
GROUP BY product_category
ORDER BY gross_revenue DESC
LIMIT 10;

-- Query 3: Geographic Revenue Density & Average order value (AOV) by Customer State
-- Business Value: Informs logistical scaling, localized marketing spend, and warehouse locations.
SELECT
	customer_state,
    COUNT(DISTINCT customer_unique_id) AS total_unique_customers,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(SUM(gross_item_value), 2) AS total_gross_spend,
    ROUND(SUM(gross_item_value) / COUNT(DISTINCT order_id), 2) AS 'average_order_value (AOV)'
FROM master_table_extended
GROUP BY customer_state
ORDER BY total_gross_spend DESC;

-- Query 4: Customer Satisfaction (Review Scores) Distribution
-- Business Value: Direct diagnostic tool for operational health and retention risks.
SELECT
	COALESCE(CAST(review_score AS CHAR), 'No Review') AS review,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(
		(COUNT(DISTINCT order_id) / (SELECT COUNT(DISTINCT order_id) FROM master_table_extended)) * 100, 2
	) AS order_percentage
FROM master_table_extended
GROUP BY review_score
ORDER BY review_score DESC;

-- Query 5: Delivery Performance vs. Customer Satisfaction
-- Business Value: Quantifies the financial and brand damage of delivery delays.
SELECT
	review_score,
	ROUND(
		AVG(DATEDIFF(order_delivered_customer_date, order_purchase_timestamp)), 1)
	AS avg_actual_delivery_days,
    ROUND(
		AVG(DATEDIFF(order_estimated_delivery_date, order_purchase_timestamp)), 1)
	AS avg_promised_delivery_days,
    ROUND(
		AVG(DATEDIFF(order_estimated_delivery_date, order_delivered_customer_date)), 1)
	AS avg_delivery_buffer_days
FROM master_table_extended
WHERE review_score IS NOT NULL AND 
	order_delivered_customer_date IS NOT NULL
GROUP BY review_score
ORDER BY review_score DESC;

-- Query 6: Top 10 High-Performing Sellers (Sales Volume & Quality)
-- Business Value: Pinpoints key merchant accounts to protect and nurture.
SELECT
	seller_id,
    COUNT(DISTINCT order_id) AS total_orders_fulfilled,
    ROUND(SUM(price), 2) AS total_merchandise_value,
    ROUND(AVG(review_score), 2) AS avg_seller_rating
FROM master_table_extended
GROUP BY seller_id
HAVING total_orders_fulfilled >= 50
ORDER BY total_merchandise_value DESC
LIMIT 10;

-- Query 7: Logistical Overhead: Freight Cost Ratio by Product Category
-- Business Value: Pinpoints categories where logistics eat up consumer purchasing power.
SELECT 
	product_category,
    ROUND(AVG(price), 2) AS avg_item_price,
    ROUND(AVG(freight_value), 2) AS avg_freight_cost,
    ROUND((AVG(freight_value) / NULLIF(AVG(price), 0)) * 100, 2) AS freight_to_price_ratio_pct
FROM master_table_extended
GROUP BY product_category
HAVING COUNT(order_id) >= 100
ORDER BY freight_to_price_ratio_pct DESC
LIMIT 15;

-- Query 8: Average Order Value (AOV) by Review Score
-- Business Value: Investigates if high-spending order cohorts hold higher expectations or worse ratings.
SELECT 
	revire_score,
    COUNT(DISTINCT order_items) AS order_count,
    ROUND(AVG(gross_item_value), 2) AS avg_gross_item_value,
    ROUND(SUM(gross_item_value), 2) AS gross_total_spend
FROM master_table_extended
WHERE review_score IS NOT NULL
GROUP BY review_score;

-- Query 9: Product Category Quality Audit (Worst Rated Categories)
-- Business Value: Surfaces defective categories causing high returns or churn.
SELECT
	product_category,
    COUNT(DISTINCT order_id) AS order_count,
    ROUND(AVG(review_score), 2) AS average_rating,
    SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) AS total_bad_reviews,
    ROUND(
		SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) / COUNT(order_id) * 100, 2
	) AS bad_review_rate_pct
FROM master_table_extended
GROUP BY product_category
HAVING total_bad_reviews >= 50
ORDER BY average_rating ASC
LIMIT 10;


-- Query 10: Repeat Purchase Behavior across Product Categories
-- Business Value: Shows which categories naturally drive customer loyalty.
WITH CategoryPurchases AS (
	SELECT 
		customer_unique_id,
        product_category,
        COUNT(DISTINCT order_id) AS order_count
	FROM master_table_extended
    GROUP BY customer_unique_id, product_category
)
SELECT 
	product_category,
    COUNT(DISTINCT customer_unique_id) AS total_category_buyer,
    SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) AS repeat_buyer,
    ROUND(
		SUM((CASE WHEN order_count > 1 THEN 1 ELSE 0 END)) / COUNT(DISTINCT customer_unique_id), 2
	) AS repeat_buyer_rate_pct
FROM CategoryPurchases
GROUP BY product_category
HAVING total_category_buyer >= 100
ORDER BY repeat_buyer_rate_pct DESC
LIMIT 10;