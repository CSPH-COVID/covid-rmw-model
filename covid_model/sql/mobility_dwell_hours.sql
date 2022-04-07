select
    measure_date
    , origin_county_id
    , destination_county_id
    , total_dwell_duration_hrs
from `co-covid-models.mobility.county_to_county_by_week`
order by 1, 2, 3