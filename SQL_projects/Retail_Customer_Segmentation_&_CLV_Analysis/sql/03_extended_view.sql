-- 03_extended_view.sql

CREATE OR REPLACE VIEW master_table_extended AS
SELECT
	-- Customer location details
    c.customer_unique_id,
    c.customer_state,
    c.customer_city,
    
    -- Order Identifiers & Fulfillment Timestemps
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    
    -- Itemized Financial Metrics
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    (oi.price + oi.freight_value) AS gross_item_value,
    
    -- Customer Feedback Score
    r.review_score,
    
    -- Product Categories 
	COALESCE(t.product_category_name_english, p.product_category_name, 'unknown_category') AS product_category
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN order_reviews r ON o.order_id = r.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
WHERE order_status = 'delivered';

SELECT * FROM master_table_extended;
