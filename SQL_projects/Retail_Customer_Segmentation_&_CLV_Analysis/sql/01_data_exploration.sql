-- 01_data_exploration.sql

-- Olist E-commerce Project: Initial Database Sanity Checks & EDA
-- Purpose: Quick audit of raw tables, checking for missing values and baseline dates.


-- Quick check on what tables loaded into the schema
SHOW TABLES;

-- Double-checking actual row volumes per table to make sure imports didn't truncate
SELECT 
    table_name, 
    table_rows AS row_count 
FROM information_schema.tables 
WHERE table_schema = 'olist_db'
ORDER BY table_rows DESC;


-- MISSING VALUE (NULL) AUDITS


-- Checking if the customers table has any blank rows or broken IDs
SELECT COUNT(*) AS broken_customer_rows
FROM customers 
WHERE customer_id IS NULL
   OR customer_unique_id IS NULL
   OR customer_zip_code_prefix IS NULL
   OR customer_city IS NULL
   OR customer_state IS NULL;

-- Checking for critical missing fields in the core transaction table
SELECT COUNT(*) AS blank_order_records
FROM orders
WHERE order_id IS NULL 
   OR customer_id IS NULL 
   OR order_status IS NULL 
   OR order_purchase_timestamp IS NULL;


-- BUSINESS RULES & DATE BOUNDARIES


-- Listing out all possible order statuses to find out what counts as an active sale
SELECT DISTINCT order_status FROM orders;

-- Finding the absolute start and end dates of the dataset to anchor our Recency calculation later
SELECT 
    MIN(order_purchase_timestamp) AS dataset_start_date,
    MAX(order_purchase_timestamp) AS dataset_end_date
FROM orders;


-- PROTOTYPING EARLY VIEWS (V1 Pipeline)


/* 
	NOTE: These are basic prototype views used during early development.
	The production-grade pipeline using advanced CTEs and window functions 
	is saved separately under `sql/01_rfm_scorecard_abt.sql`.
*/

-- Joining core sales data into a quick flat table for baseline testing
CREATE OR REPLACE VIEW prototype_master_table_itemized AS
SELECT 
    c.customer_unique_id,
    o.order_id,
    o.order_purchase_timestamp,
    oi.product_id,
    oi.price,
    oi.freight_value,
    p.payment_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN order_payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered';

-- Viewing the prototype master table
SELECT * FROM prototype_master_table_itemized;
 
-- Building a fast, unoptimized RFM summary to confirm calculation logic
CREATE OR REPLACE VIEW prototype_rfm_aggregation AS
SELECT 
    customer_unique_id,
    DATEDIFF(
        (SELECT MAX(order_purchase_timestamp) FROM prototype_master_table_itemized), 
        MAX(order_purchase_timestamp)
    ) AS recency,
    COUNT(DISTINCT order_id) AS frequency,
    SUM(payment_value) AS monetary
FROM prototype_master_table_itemized
GROUP BY customer_unique_id;

-- Viewing the prototype master table
SELECT * FROM prototype_rfm_aggregation;