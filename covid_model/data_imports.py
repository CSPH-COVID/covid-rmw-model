""" Python Standard Library """
import datetime as dt
import json
""" Third Party Imports """
import numpy as np
import pandas as pd
import pickle
""" Local Imports """


def normalize_date(date):
    """Convert datetime to date if necessary

    Args:
        date: either a dt.datetime.date or dt.datetime object

    Returns: dt.datetime.date object

    """
    return date if type(date) == dt.date or date is None else date.date()


class ExternalData:
    """Base class for loading external data, either from file or database

    """
    def __init__(self, engine=None, t0_date=None, fill_from_date=None, fill_to_date=None):
        self.engine = engine
        self.t0_date = normalize_date(t0_date) if t0_date is not None else None
        self.fill_from_date = normalize_date(fill_from_date) if fill_from_date is not None else normalize_date(t0_date)
        self.fill_to_date = normalize_date(fill_to_date) if fill_to_date is not None else None

    def fetch(self, fpath=None, rerun=True, **args):
        """Template function for retrieving data and optionally saving to a file

        Args:
            fpath: optional location to save data
            rerun: whether to fetch the data again
            **args: additional arguments passed to self.fetch_from_db

        Returns:

        """
        if rerun:
            df = self.fetch_from_db(**args)
            if fpath is not None:
                df.reset_index().drop(columns='index', errors='ignore').to_csv(fpath, index=False)
        else:
            df = pd.read_csv(fpath)

        if self.t0_date is not None:
            index_names = [idx for idx in df.index.names if idx not in (None, 'measure_date')]
            df = df.reset_index()
            df['t'] = (pd.to_datetime(df['measure_date']).dt.date - self.t0_date).dt.days
            min_t = min(df['t'])
            max_t = max(df['t'])
            df = df.reset_index().drop(columns=['index', 'level_0', 'measure_date'], errors='ignore').set_index(['t'] + index_names)

            trange = range((self.fill_from_date - self.t0_date).days, (self.fill_to_date - self.t0_date).days + 1 if self.fill_to_date is not None else max_t)
            index = pd.MultiIndex.from_product([trange] + [df.index.unique(level=idx) for idx in index_names]).set_names(['t'] + index_names) if index_names else range(max_t)
            empty_df = pd.DataFrame(index=index)
            df = empty_df.join(df, how='left').fillna(0)

        return df

    def fetch_from_db(self, **args) -> pd.DataFrame:
        """fetch data from database and return Pandas Dataframe

        Args:
            **args: arguments for pandas.read_sql function

        Returns: pandas dataframe of loaded data

        """
        # return pd.read_sql(args['sql'], self.engine)
        return pd.read_sql(con=self.engine, **args)


class ExternalRegionDefs(ExternalData):
    """Class for retrieving vaccination projection parameters from the database.

    """
    def fetch_from_db(self, **args) -> pd.DataFrame:
        """Retrieve vaccination projection parameters from the database."""
        with open("covid_model/sql/region_defs.sql","r") as sql:
            df = pd.read_sql(sql.read(),
                             self.engine,
                             index_col="region_id")
        return df


class ExternalPopulation(ExternalData):
    """Class for retrieving population data from the database.

    """
    def fetch_from_db(self) -> pd.DataFrame:
        """Retrieve population data from database using query in """
        with open("covid_model/sql/cste_population.sql","r") as sql:
            df = pd.read_sql(sql.read(),
                             self.engine,
                             index_col="region_id")\
                .rename(columns={"n1":"0-17","n2":"18-64","n3":"65+"})

        # Verify that population values match by up across each row (region) and then check that the sum is 2x the
        # value of the region_pop column. Since the other age group columns should add up to region_pop, any
        # discrepancies mean there is a data issue.
        bad_regions = list(df.index[df.sum(axis=1) != 2*df["region_pop"]])
        if len(bad_regions) != 0:
            raise RuntimeError(f"Regional population does not match population total for regions {bad_regions}.")
        return df


class ExternalVariantProportions(ExternalData):
    """Class to retrieve external variant proportion data from the database. Used to align variant proportions.

    """
    def fetch_from_db(self, region_id) -> pd.DataFrame:
        # with open("covid_model/input/test_variant_props.pkl","rb") as f:
        #     raw_df = pickle.load(f)
        #     raw_df.set_index(pd.date_range(start="2020-01-24",periods=len(raw_df),freq="D"),inplace=True)
        # inf_df = raw_df["I"].groupby("variant",axis=1).sum()
        # prop_df = inf_df.divide(inf_df.sum(axis=1),axis="index")
        # prop_df.loc["2020-01-24", "wildtype"] = 1.0
        # prop_df.fillna(0,inplace=True)
        sql = open("covid_model/sql/variant_proportions.sql","r").read()
        raw_df = pd.read_sql(sql=sql,
                             con=self.engine,
                             index_col=["date"],
                             parse_dates=["date"],
                             params={"region_id": region_id}).drop(columns=["state"])
        raw_df["none"] = 0.0
        return raw_df


class ExternalHospsEMR(ExternalData):
    """Class for Retrieving EMResource hospitalization data from database

    """
    def fetch_from_db(self):
        """Retrieve hospitalization data from database using query in emresource_hospitalizations.sql

        Returns: Pandas dataframe of hospitalization data

        """
        sql = open('covid_model/sql/emresource_hospitalizations.sql', 'r').read()
        return pd.read_sql(sql, self.engine, index_col=['measure_date'])


class ExternalHospsCOPHS(ExternalData):
    """Class for Retrieving COPHS hospitalization data from database

    """
    def fetch_from_db(self, region_ids: list):
        """Retrieve hospitalization data from database using query in hospitalized_county_subset.sql

        COPHS contains county level hospitalizations, so optionally you can specify a subset of counties to query for.

        Args:
            county_ids: list of county FIPS codes that you want hospitalization data for

        Returns: Pandas dataframe of hospitalization data

        """
        sql = open('covid_model/sql/hospitalized_region_subset.sql', 'r').read()
        return pd.read_sql(sql, self.engine, index_col=['measure_date'], params={'region_ids': region_ids})


class ExternalVacc(ExternalData):
    """Class for retrieving vaccinations data from database

    """
    def fetch_from_db(self, region_id: str = None, county_ids: list = None):
        """Retrieve vaccinations from database using query in sql file, either for entire state or for a subset of counties

        Args:
            region_id: region id (e.g. con, nme, etc) to fetch vaccinations for (optional)

        Returns:
            A Dataframe containing the vaccination data.
        """
        if region_id is None:
            sql = open('covid_model/sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()
            return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
        else:
            sql = open("covid_model/sql/vaccination_by_region_by_age_group.sql","r").read()
            return pd.read_sql(sql,self.engine,index_col=["measure_date","age"],params={"region_id":region_id})
            #sql = open("covid_model/sql/vaccination_by_age_group_with_boosters_wide.sql","r").read()
            #return pd.read_sql(sql,self.engine, index_col=["measure_date","age"])
            # This query passes in region/county IDs which can be used to subset the data (Should only be used if the
            # population scaling is turned OFF).
            #sql = open("covid_model/sql/vaccination_by_age_with_boosters.sql","r").read()
            #return pd.read_sql(sql, self.engine, index_col=["measure_date","age"], params={"county_ids": county_ids})

            # Old
            #sql = open('covid_model/sql/vaccination_by_age_group_with_boosters_wide.sql', 'r').read()
            #return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'])
            # Older
            #sql = open('covid_model/sql/vaccination_by_age_group_with_boosters_wide_county_subset.sql', 'r').read()
            #return pd.read_sql(sql, self.engine, index_col=['measure_date', 'age'], params={'county_ids': county_ids})

def get_region_mobility_from_db(engine, county_ids=None, fpath=None) -> pd.DataFrame:
    """Standalone function to retrieve mobility data from database and possibly write to a file

    Args:
        engine: connection to database
        county_ids: list of FIPS codes to retrieve mobility data for
        fpath: file path to save the mobility data once downloaded (optional)

    Returns:

    """
    if county_ids is None:
        with open('covid_model/sql/mobility_dwell_hours.sql') as f:
            df = pd.read_sql(f.read(), engine, index_col=['measure_date'])
    else:
        with open('covid_model/sql/mobility_dwell_hours_county_subset.sql') as f:
            df = pd.read_sql(f.read(), engine, index_col=['measure_date'], params={'county_ids': county_ids})
    if fpath:
        df.to_csv(fpath)
    return df
