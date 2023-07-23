SELECT vp.*
FROM `co-covid-models.cste.variant_proportions` vp
JOIN `co-covid-models.cste.state_region_map` mp ON mp.state_name = vp.state
WHERE mp.region_id = %(region_id)s
