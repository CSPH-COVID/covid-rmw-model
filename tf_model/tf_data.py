import json
import pandas as pd
import numpy as np
import datetime as dt
from covid_model.data_imports import ExternalVacc, ExternalPopulation
from sqlalchemy.engine import create_engine

class ModelParameters:

    def __init__(self, start_date=dt.datetime.strptime("2020-01-01","%Y-%m-%d").date(), end_date=dt.datetime.strptime("2023-06-30","%Y-%m-%d").date(), region="con"):

        self.start_date = start_date
        self.end_date = end_date
        self.tstart = 0
        self.tend = (self.end_date - self.start_date).days + 1
        self.daterange = pd.date_range(self.start_date, end=self.end_date).date
        self.params = []
        self.vacc_proj_params = {}
        self.region = region
        self.actual_vacc_df = None
        self.proj_vacc_df = None
        self.population_df = None

    @property
    def all_vacc_df(self):
        return pd.concat([self.actual_vacc_df, self.proj_vacc_df],axis=0)

    def date_to_t(self, date):
        """Convert a date (string or date object) to t, number of days since model start date.

        Args:
            date: either a string in the format 'YYYY-MM-DD' or a date object.

        Returns: integer t, number of days since model start date.

        """
        if isinstance(date, str):
            return (dt.datetime.strptime(date, "%Y-%m-%d").date() - self.start_date).days
        else:
            return (date - self.start_date).days

    def t_to_date(self, t):
        """Convert a t, number of days since model start date, to a date object.

        Args:
            t: number of days since model start date.

        Returns: date object representing the t in question.

        """
        return self.start_date + dt.timedelta(days=t)

    def load_all_params(self):
        self.params.clear()
        # Create an engine to fetch new data
        engine = create_engine("bigquery://co-covid-models")

        # Vacc projection params
        self.load_vacc_proj_params()

        # Parameters from file
        self.load_params_file()

        # Region Population Parameters
        self.load_region_population(engine=engine)

        # Vaccination Data
        self.load_and_compute_vacc_data(engine=engine)

        # TC data
        self.load_tc_params()

        print("All parameters loaded.")

    def load_tc_params(self, filename="tf_model/tc_temp.json"):
        with open(filename,"r") as f:
            tc_file = json.load(f)
        temp_vals = {self.t_to_date(int(t)).strftime("%Y-%m-%d"): d["con"] for t, d in tc_file.items()}
        param = {"param": "TC",
                 "attrs": None,
                 "vals": {"2020-01-01": 1.0}}
        self.params.append(param)

    def load_vacc_proj_params(self):
        # Clear old params
        self.vacc_proj_params.clear()
        # Load new params
        with open("covid_model/input/rmw_vacc_proj_params.json","r") as f:
            vpp = json.load(f)
        # Add to list
        self.vacc_proj_params.update(vpp)

    def load_params_file(self):
        # Clear old params
        self.params.clear()
        # Load new params
        with open("covid_model/input/rmw_temp_params_changes_test.json", "r") as f:
            new_params = json.load(f)
        # Add new parameters to list
        self.params.extend(new_params)

    def load_and_compute_vacc_data(self, engine, vacc_delay=14):
        self.load_actual_vacc_data(engine=engine)
        self.generate_vacc_projections()

        vacc_per_available = self.vacc_per_available()

        # apply vacc_delay
        vacc_per_available = vacc_per_available.groupby(['region', 'age']).shift(vacc_delay).fillna(0)

        # group vacc_per_available by trange interval
        # bins = self.params_trange + [self.tend] if self.tend not in self.params_trange else self.params_trange
        bins = list(range(self.tstart, self.tend, 7))
        bins = bins + [self.tend] if self.tend not in bins else bins
        t_index_rounded_down_to_tslices = pd.cut(vacc_per_available.index.get_level_values('t'), bins, right=False,
                                                 retbins=False, labels=bins[:-1])
        vacc_per_available = vacc_per_available.groupby([t_index_rounded_down_to_tslices, 'region', 'age']).mean()
        vacc_per_available['date'] = [self.t_to_date(d) for d in vacc_per_available.index.get_level_values(0)]
        vacc_per_available = vacc_per_available.reset_index().set_index(['region', 'age']).sort_index()

        # set the fail rate and vacc per unvacc rate for each dose
        for shot in ["shot1","shot2","booster1","booster23"]:
            for age in ["0-17","18-64","65+"]:
                    vpa_sub = vacc_per_available[['date', shot]].loc[self.region, age]
                    # TO DO: hack for when there's only one date. Is there a better way?
                    if isinstance(vpa_sub, pd.Series):
                        vpa_sub = {vpa_sub[0]: vpa_sub[1]}
                    else:
                        vpa_sub = vpa_sub.reset_index(drop=True).set_index('date').sort_index().drop_duplicates().to_dict()[shot]
                    self.params.append({"param":f"{shot}_per_available",
                                        "attrs": {"age": age},
                                        "vals": {k.strftime("%Y-%m-%d"): v for k,v in vpa_sub.items()}})
                    # self.set_compartment_param(param=f'{shot}_per_available', attrs={'age': age, 'region': region},
                    #                            vals=vpa_sub)

    def load_actual_vacc_data(self, engine):
        def decumulative(g):
            # Run diff(), but keep the first row of data, which is set to NaN by the .diff() function.
            tmp_g = g.diff()
            tmp_g.iloc[0] = g.iloc[0]
            return tmp_g

        actual_vacc_df_list = []
        # for region in self.regions:
        tmp_vacc = ExternalVacc(engine).fetch(region_id=self.region).set_index('region', append=True).reorder_levels(
            ['measure_date', 'region', 'age'])
        # Check that vaccinations don't exceed population for any regions.
        vacc_data_sum = tmp_vacc.groupby(["region", "age"]).sum()
        # Get the population values for each region/age group combination.
        pop_raw_vals = pd.DataFrame.from_dict({ p["attrs"]["age"]: p["vals"]["2020-01-01"]
                                               for p in self.params if p["param"] == "region_age_pop"},
                                              orient="index").squeeze()
        #pop_raw_vals.index = pd.MultiIndex.from_tuples(pop_raw_vals.index, names=["region", "age"])
        # Maximum percentages of population age groups which should have received vaccine.
        vacc_thresh_pct = pd.DataFrame.from_dict(self.vacc_proj_params["max_cumu"])
        vacc_thresh_pct.index.name = "age"
        # Calculate the maximum number of people within each region/age_group combination who should have received
        # the vaccine.
        vacc_thresh_abs = vacc_thresh_pct.multiply(pop_raw_vals, axis="index", level="age")
        # Check that these threshold numbers are above what we see in the vaccination data.
        vacc_over_thresh = vacc_data_sum.gt(vacc_thresh_abs, axis="index")
        idx_gt, col_gt = np.where(vacc_over_thresh)
        # If we found issues, stop here and raise error
        if len(idx_gt) != 0:
            bad_idcs = list(zip(vacc_over_thresh.index[idx_gt], vacc_over_thresh.columns[col_gt]))
            #bad_idcs_str = "\n".join([str(x) for x in bad_idcs])
            raise RuntimeError(f"Error: Some vaccinations are over thresholds! {bad_idcs}")
        # Check that vaccination data is consistent
        if tuple(tmp_vacc.groupby(["region", "age"]).cumsum().idxmax(axis=1).unique()) != ("shot1",):
            # TODO: If we ever change it so that shots can happen out of sequence, we will need to remove this.

            # Transform the data to cumulative
            tmp_vacc_cum = tmp_vacc.groupby(["region", "age"]).cumsum()
            # Take the element-wise minimum of the current cumulative data and the data shifted to the right
            # by one column (i.e. the previous dose's cumulative #). Fill the first column with np.inf to ensure
            # that the original first column is preserved.
            tmp_vacc_cum = np.minimum(tmp_vacc_cum, tmp_vacc_cum.shift(axis=1).fillna(np.inf))
            tmp_vacc = tmp_vacc_cum.groupby(["region", "age"]).apply(decumulative)

        actual_vacc_df_list.append(tmp_vacc)

        self.actual_vacc_df = pd.concat(actual_vacc_df_list)
        self.actual_vacc_df.index.set_names('date', level=0, inplace=True)

    def generate_vacc_projections(self):
        """Create projections for vaccines to fill in any gaps between actual vaccinations and the model end_date

        This method relies on the vacc_proj_params to specify how projections should be made.

        """
        proj_lookback = self.vacc_proj_params['lookback']
        proj_fixed_rates = self.vacc_proj_params['fixed_rates']
        max_cumu = self.vacc_proj_params['max_cumu']
        max_rate_per_remaining = self.vacc_proj_params['max_rate_per_remaining']
        realloc_priority = self.vacc_proj_params['realloc_priority']

        shots = list(self.actual_vacc_df.columns)
        region_df = pd.DataFrame({'region': [self.region]})

        # add projections
        proj_from_date = self.actual_vacc_df.index.get_level_values('date').max() + dt.timedelta(days=1)
        proj_to_date = self.end_date
        if proj_to_date >= proj_from_date:
            proj_date_range = pd.date_range(proj_from_date, proj_to_date).date
            # project daily vaccination rates based on the last {proj_lookback} days of data
            projected_rates = self.actual_vacc_df[self.actual_vacc_df.index.get_level_values(0) >= proj_from_date - dt.timedelta(days=proj_lookback)].groupby(['region', 'age']).sum() / proj_lookback
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                proj_fixed_rates_df = pd.DataFrame(proj_fixed_rates).rename_axis(index='age').reset_index().merge(region_df, how='cross').set_index(['region', 'age'])
                for shot in shots:
                    # Note: currently treats all regions the same. Need to change if finer control desired
                    projected_rates[shot] = proj_fixed_rates_df[shot]
            # build projections
            projections = pd.concat({d: projected_rates for d in proj_date_range}).rename_axis(index=['date', 'region', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                cumu_vacc = self.actual_vacc_df.groupby(['region', 'age']).sum()
                groups = realloc_priority if realloc_priority else projections.groupby(['region', 'age']).sum().index
                # self.params_by_t hasn't necessarily been built yet, so use a workaround
                populations = pd.DataFrame([{"region": self.region, 'age': param_dict['attrs']['age'], 'population': list(param_dict['vals'].values())[0]} for param_dict in self.params if param_dict['param'] == 'region_age_pop'])

                for d in projections.index.unique('date'):
                    # Note: I simplified this, so max_cumu can't vary in time. wasn't being used anyways, and it used the old 'tslices' paradigm (-Alex Fout)
                    this_max_cumu = max_cumu.copy()

                    # Note: currently treats all regions the same. Need to change if finer control desired
                    max_cumu_df = pd.DataFrame(this_max_cumu).rename_axis(index='age').reset_index().merge(region_df, how='cross').set_index(['region', 'age']).sort_index()
                    max_cumu_df = max_cumu_df.mul(pd.DataFrame(populations).set_index(['region', 'age'])['population'], axis=0)
                    for i in range(len(groups)):
                        group = groups[i]
                        key = tuple([d] + list(group))
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        # Limit the rates so that later shot cumulative values never exceed earlier shot cumulative
                        # values.
                        max_rate = max_rate.clip(upper=(-cumu_vacc.loc[group].diff(periods=1)).fillna(np.inf))
                        excess_rate = (projections.loc[key] - max_rate).clip(lower=0)
                        projections.loc[key] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[tuple([d] + list(groups[i + 1]))] += excess_rate

                    cumu_vacc += projections.loc[d]

            self.proj_vacc_df = projections
        else:
            self.proj_vacc_df = None

    def vacc_per_available(self):
        # Construct a cleaned version of how many of each shot are given on each day to each age group in each region
        vacc_rates = self.all_vacc_df
        missing_dates = pd.DataFrame({'date': [d for d in self.daterange if d < min(vacc_rates.index.get_level_values('date'))]})
        missing_shots = missing_dates.merge(pd.DataFrame(index=vacc_rates.reset_index('date').index.unique(), columns=vacc_rates.columns).fillna(0).reset_index(), 'cross').set_index(['date', 'region', 'age'])
        vacc_rates = pd.concat([missing_shots, vacc_rates])
        vacc_rates['t'] = [self.date_to_t(d) for d in vacc_rates.index.get_level_values('date')]
        vacc_rates = vacc_rates.set_index('t', append=True)
        # get the population of each age group (in each region) at each point in time. Should work with changing populations, but right now pop is static
        # also, attrs doesn't matter since the `region_age_pop` is just specific to an age group and region
        #populations = self.get_param_for_attrs_by_t('region_age_pop', attrs={'vacc': 'none', 'variant': 'none', 'immun': 'low'}).reset_index([an for an in self.param_attr_names if an not in ['region', 'age']])[['region_age_pop']]
        populations = self.population_df
        vacc_rates_ts = vacc_rates.index.get_level_values('t').unique()
        populations = populations.iloc[[t in vacc_rates_ts for t in populations.index.get_level_values('t')]].reorder_levels(['region', 'age', 't']).sort_index()
        populations.rename(columns={'region_age_pop': 'population'}, inplace=True)
        # compute the cumulative number of each shot to each age group in each region
        cumu_vacc = vacc_rates.sort_index().groupby(['region', 'age']).cumsum()
        # compute how many people's last shot is shot1, shot2, etc. by subtracting shot1 from shot2, etc.
        cumu_vacc_final_shot = cumu_vacc - cumu_vacc.shift(-1, axis=1).fillna(0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.join(populations)
        # compute how many people have had no shots
        # vaccinations eventually overtake population (data issue) which would make 'none' < 0 so clip at 0
        cumu_vacc_final_shot['none'] = (cumu_vacc_final_shot['population'] * 2 - cumu_vacc_final_shot.sum(axis=1)).clip(lower=0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.drop(columns='population')
        cumu_vacc_final_shot = cumu_vacc_final_shot.reindex(columns=['none', 'shot1', 'shot2', 'booster1', 'booster23'])
        # compute what fraction of the eligible population got each shot on a given day.
        available_for_vacc = cumu_vacc_final_shot.shift(1, axis=1).drop(columns='none')
        vacc_per_available = (vacc_rates / available_for_vacc).fillna(0).replace(np.inf, 0).reorder_levels(['t', 'date', 'region', 'age']).sort_index()
        # because vaccinations exceed the population, we can get rates greater than 1. To prevent compartments have negative people, we have to cap the rate at 1
        vacc_per_available = vacc_per_available.clip(upper=1)
        return vacc_per_available

    def load_region_population(self, engine):
        # Get the DataFrame of populations
        external_pop_df = ExternalPopulation(engine).fetch()
        # Convert to a list of records so we can iterate
        external_pop_records = external_pop_df.melt(ignore_index=False).to_records()
        # Create a new dictionary and update param_defs.
        pop_params = []
        for region, age, pop in external_pop_records:
            if region != self.region:
                continue
            is_reg_age_pop = (age != "region_pop")
            tmp_d = {"param": "region_age_pop" if is_reg_age_pop else "region_pop",
                     "attrs": None,
                     "vals": {"2020-01-01": pop}}
            if is_reg_age_pop:
                if tmp_d["attrs"] is None:
                    tmp_d["attrs"] = {}
                tmp_d["attrs"]["age"] = age
            pop_params.append(tmp_d)
        self.params.extend(pop_params)

        # Make this weird dataframe we need for getting vaccination data
        tmp_external_pop_df = external_pop_df.drop(columns="region_pop").stack()
        regions,ages = tmp_external_pop_df.index.levels
        days = range((self.end_date - self.start_date).days+1)
        new_index = pd.MultiIndex.from_product([regions,ages,days])
        self.population_df = tmp_external_pop_df.repeat(len(days)).to_frame().set_index(new_index)
        self.population_df.index.names = ["region","age","t"]
        self.population_df = self.population_df.reorder_levels(["t","age","region"])
        self.population_df.rename(columns={0:"region_age_pop"},inplace=True)

if __name__ == "__main__":
    mp = ModelParameters()
    mp.load_all_params()
