SELECT 
	order_id,
	order_purchase_timestamp,
EXTRACT(MONTH FROM order_purchase_timestamp ) AS purchase_month, 
EXTRACT(YEAR FROM order_purchase_timestamp) AS purchase_year

FROM orders;
--ajouter la colonne des mois
ALTER TABLE orders
add column if not exists purchase_month int;
Update orders
set purchase_month=EXTRACT(MONTH FROM order_purchase_timestamp);

--ajouter la colonne des annees
ALTER TABLE orders
add column if not exists purchase_year int;
Update orders
set purchase_year=EXTRACT(YEAR FROM order_purchase_timestamp);

--ajouter la colonne is_canceled
ALTER TABLE orders
add column if not exists is_canceled INT;

UPDATE orders
set is_canceled=
	CASE
		WHEN order_status='canceled' THEN 1
		ELSE 0
	END;
-- select * from orders
-- limit 10;
SELECT *
FROM public.orders
where is_canceled=1;
