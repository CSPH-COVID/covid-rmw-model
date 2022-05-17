### Python Standard Library ###
import json
import math
import datetime as dt
import copy
from operator import itemgetter
from collections import OrderedDict, defaultdict
import logging
### Third Party Imports ###
import numpy as np
import pandas as pd
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import scipy.integrate as spi
import scipy.sparse as spsp
from sqlalchemy import func
from sqlalchemy.orm import Session
from sortedcontainers import SortedDict
### Local Imports ###
from covid_model.db import get_sqa_table
from covid_model.ode_flow_terms import ConstantODEFlowTerm, ODEFlowTerm
from covid_model.data_imports import ExternalVacc, ExternalHosps, get_region_mobility_from_db
from covid_model.utils import get_params, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})


# class used to run the model given a set of parameters, including transmission control (ef)
class CovidModel:
    ####################################################################################################################
    ### Setup
    __tmin = 0

    ####################################################################################################################
    ### Initialization and Updating

    def __init__(self, engine=None, base_model=None, update_derived_properties=True, base_spec_id=None, **margs):
        # margs can be any model property

        # basic model data
        self.attrs = OrderedDict({'seir': ['S', 'E', 'I', 'A', 'Ih', 'D'],
                             'age': ['0-19', '20-39', '40-64', '65+'],
                             'vacc': ['none', 'shot1', 'shot2', 'shot3'],
                             'priorinf': ['none', 'non-omicron', 'omicron'],
                             'variant': ['none', 'alpha', 'delta', 'omicron', 'ba2'],
                             'immun': ['none', 'weak', 'strong'],
                             'region': ['co']})

        self.tags = {}

        self.__start_date = dt.datetime.strptime("2020-01-01", "%Y-%m-%d").date()
        self.__end_date = dt.datetime.strptime("2023-01-01", "%Y-%m-%d").date()

        self.tc_tslices = list()
        self.tc = list()
        self.tc_cov = None

        self.solution = None

        # model data
        self.__params_defs = None
        self.__region_defs = None
        self.regions = self.attrs['region']
        self.__vacc_proj_params = None
        self.__mobility_mode = None
        self.actual_mobility = {}
        self.mobility_proj_params = None
        self.actual_vacc_df = None
        self.proj_vacc_df = None
        self.recently_updated_properties = []
        self.actual_hosp = None

        self.base_spec_id = None
        self.spec_id = None
        self.region_fit_spec_ids = None
        self.region_fit_result_ids = None

        # ode data
        self.t_prev_lookup = None
        self.terms = None
        self.compartments_as_index = None
        self.compartments = None
        self.param_compartments = None
        self.params_by_t = None
        self.cmpt_idx_lookup = list()

        self.linear_matrix = None
        self.nonlinear_matrices = None
        self.constant_vector = None
        self.nonlinear_multiplier = {}
        self.max_step_size = np.inf


        if base_model is not None and base_spec_id is not None:
            ermsg = "Cannot pass both a base_model and base_spec_id"
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise ValueError(ermsg)

        # if there is a base model, take all its properties
        if base_model is not None:
            logger.debug(f"{str(self.tags)} Copying from base model")
            for key, val in vars(base_model).items():
                setattr(self, key, copy.deepcopy(val))

        # if a base_spec_id provided, load from the database
        if base_spec_id is not None:
            logger.debug(f"{str(self.tags)} Copying base specifications")
            self.base_spec_id = base_spec_id
            self.read_from_base_spec_id(engine)

        # update any attributes with items in **margs
        if len(margs.keys()) > 0:
            logger.debug(f"{str(self.tags)} Applying model arguments")
        for key, val in margs.items():
            setattr(self, key, copy.deepcopy(val))
            self.recently_updated_properties.append(key)

        if update_derived_properties:
            self.update(engine)

    # some properties are derived / computed / constructed from other properties. If the non-derived
    # properties are updated, the derived properties may need recomputing
    def update(self, engine):
        if any([p in self.recently_updated_properties for p in ['regions', 'region_defs']]):
            logger.debug(f"{str(self.tags)} Updating actual vaccines")
            self.set_actual_vacc(engine)

        if any([p in self.recently_updated_properties for p in ['end_date', 'vacc_proj_params']]) and self.vacc_proj_params is not None:
            logger.debug(f"{str(self.tags)} Updating Projected Vaccines")
            self.set_proj_vacc()

        if any([p in self.recently_updated_properties for p in ['end_date', 'mobility_mode']]) and self.mobility_mode is not None:
            logger.debug(f"{str(self.tags)} Updating Actual Mobility")
            self.set_actual_mobility(engine)

        if any([p in self.recently_updated_properties for p in ['end_date', 'mobility_mode', 'mobility_proj_params']]) and self.mobility_mode is not None:
            logger.debug(f"{str(self.tags)} Updating Projected Mobility")
            self.set_proj_mobility()

        if any([p in self.recently_updated_properties for p in ['end_date', 'model_mobility_mode', 'mobility_proj_params']]):
            if self.mobility_mode is not None and self.mobility_mode != "none":
                logger.debug(f"{str(self.tags)} Getting Mobility As Parameters")
                self.params_defs.update(self.get_mobility_as_params())

        if any([p in self.recently_updated_properties for p in ['region_fit_spec_ids', 'region_fit_result_ids']]):
            logger.debug(f"{str(self.tags)} Getting kappas as parameters using region fits")
            self.params_defs.update(self.get_kappas_as_params_using_region_fits(engine))

        if any([p in self.recently_updated_properties for p in ['tc', 'tc_tslices']]):
            logger.debug(f"{str(self.tags)} Updating tc_t_prev_lookup and applying TC")
            self.tc_t_prev_lookup = {t_int: max(t for t in [0] + self.tc_tslices if t <= t_int) for t_int in self.trange}
            self.apply_tc(force_nlm_update=True)

        if any([p in self.recently_updated_properties for p in ['start_date', 'regions', 'region_defs']]):
            logger.debug(f"{str(self.tags)} Setting Actual Hospitalizations")
            self.set_actual_hosp(engine)

        self.recently_updated_properties = []

    ####################################################################################################################
    ### Functions to Update Derived Properites and Retrieve Data

    def set_actual_vacc(self, engine, actual_vacc_df=None):
        if engine is not None:
            logger.debug(f"{str(self.tags)} getting vaccines from db")
            actual_vacc_df_list = []
            for region in self.regions:
                county_ids = self.region_defs[region]['counties_fips']
                actual_vacc_df_list.append(ExternalVacc(engine).fetch(county_ids=county_ids).assign(region=region).set_index('region', append=True).reorder_levels(['measure_date', 'region', 'age']))
            self.actual_vacc_df = pd.concat(actual_vacc_df_list)
            self.actual_vacc_df.index.set_names('date', level=0, inplace=True)
        if actual_vacc_df is not None:
            logger.debug(f"{str(self.tags)} copying vaccines from file")
            self.actual_vacc_df = actual_vacc_df.copy()
        if actual_vacc_df is None and engine is None:
            logger.warning(f"{str(self.tags)} no engine or vacc file provided, not updating vaccines")

    def set_proj_vacc(self):
        proj_lookback = self.vacc_proj_params['lookback']
        proj_fixed_rates = self.vacc_proj_params['fixed_rates']
        max_cumu = self.vacc_proj_params['max_cumu']
        max_rate_per_remaining = self.vacc_proj_params['max_rate_per_remaining']
        realloc_priority = self.vacc_proj_params['realloc_priority']

        shots = list(self.actual_vacc_df.columns)
        region_df = pd.DataFrame({'region': self.regions})

        # add projections
        proj_from_date = self.actual_vacc_df.index.get_level_values('date').max() + dt.timedelta(days=1)
        proj_to_date = self.end_date
        if proj_to_date >= proj_from_date:
            proj_date_range = pd.date_range(proj_from_date, proj_to_date).date
            # project daily vaccination rates based on the last {proj_lookback} days of data
            projected_rates = self.actual_vacc_df[self.actual_vacc_df.index.get_level_values(0) >= proj_from_date-dt.timedelta(days=proj_lookback)].groupby(['region', 'age']).sum()/proj_lookback
            # override rates using fixed values from proj_fixed_rates, when present
            if proj_fixed_rates:
                proj_fixed_rates_df = pd.DataFrame(proj_fixed_rates).rename_axis(index='age').reset_index().merge(region_df,how='cross').set_index(['region', 'age'])
                for shot in shots:
                    # Note: currently treats all regions the same. Need to change if finer control desired
                    projected_rates[shot] = proj_fixed_rates_df[shot]
            # build projections
            projections = pd.concat({d: projected_rates for d in proj_date_range}).rename_axis(index=['date', 'region', 'age'])

            # reduce rates to prevent cumulative vaccination from exceeding max_cumu
            if max_cumu:
                cumu_vacc = self.actual_vacc_df.groupby(['region', 'age']).sum()
                groups = realloc_priority if realloc_priority else projections.groupby(['region','age']).sum().index
                populations = pd.DataFrame([{'param': param_dict['param'], 'region': region, 'population': list(param_dict['vals'].values())[0]} for region in self.regions for param_dict in self.params_defs if region in param_dict['param']])
                populations['age'] = ['65+' if li[1]=='65p' else '-'.join(li[1:3]) for li in populations['param'].str.split("_")]
                populations = populations[populations['age'] != 'pop'].drop(columns='param')


                for d in projections.index.unique('date'):
                    this_max_cumu = get_params(max_cumu.copy(), d)

                    # Note: currently treats all regions the same. Need to change if finer control desired
                    max_cumu_df = pd.DataFrame(this_max_cumu).rename_axis(index='age').reset_index().merge(region_df, how='cross').set_index(['region', 'age']).sort_index()
                    max_cumu_df = max_cumu_df.mul(pd.DataFrame(populations).set_index(['region', 'age'])['population'], axis=0)
                    for i in range(len(groups)):
                        group = groups[i]
                        key = tuple([d] + list(group))
                        current_rate = projections.loc[key]
                        max_rate = max_rate_per_remaining * (max_cumu_df.loc[group] - cumu_vacc.loc[group])
                        excess_rate = (projections.loc[key] - max_rate).clip(lower=0)
                        projections.loc[key] -= excess_rate
                        # if a reallocate_order is provided, reallocate excess rate to other groups
                        if i < len(groups) - 1 and realloc_priority is not None:
                            projections.loc[tuple([d] + list(groups[i + 1]))] += excess_rate

                    cumu_vacc += projections.loc[d]

            self.proj_vacc_df = projections
        else:
            self.proj_vacc_df = None

    def set_actual_mobility(self, engine):
        regions = self.regions
        county_ids = [fips for region in regions for fips in self.region_defs[region]['counties_fips']]
        df = get_region_mobility_from_db(engine, county_ids=county_ids).reset_index('measure_date')

        # add regions to dataframe
        regions_lookup = {fips: region for region in regions for fips in self.region_defs[region]['counties_fips']}
        df['origin_region'] = [regions_lookup[id] for id in df['origin_county_id']]
        df['destination_region'] = [regions_lookup[id] for id in df['destination_county_id']]
        df['t'] = (df['measure_date'] - self.start_date).dt.days
        # find most recent data before self.start_date and set its time to zero so we have an initial mobility
        df.replace({'t': max(df['t'][df['t']<=0])}, 0, inplace=True)

        df = df[df['t']>=0].drop(columns=['origin_county_id', 'destination_county_id']) \
            .groupby(['t', 'origin_region', 'destination_region']) \
            .aggregate(total_dwell_duration_hrs=('total_dwell_duration_hrs', 'sum'))

        # Create dictionaries of matrices, both D and M.
        ts = df.index.get_level_values('t')
        region_idx = {region: i for i, region in enumerate(regions)}
        dwell_matrices = {}
        for t in ts:
            dfsub = df.loc[df.index.get_level_values('t') == t].reset_index('t', drop=True).reset_index()
            idx_i = [region_idx[region] for region in dfsub['origin_region']]
            idx_j = [region_idx[region] for region in dfsub['destination_region']]
            vals = dfsub['total_dwell_duration_hrs']
            dwell = spsp.coo_array((vals, (idx_i, idx_j)), shape=(len(regions), len(regions))).todense()
            dwell[np.isnan(dwell)] = 0
            dwell_rownorm = dwell / dwell.sum(axis=1)[:, np.newaxis]
            dwell_colnorm = dwell / dwell.sum(axis=0)[np.newaxis, :]
            dwell_matrices[t] = {"dwell": dwell.tolist(), "dwell_rownorm": dwell_rownorm.tolist(), "dwell_colnorm": dwell_colnorm.tolist()}

        self.actual_mobility = dwell_matrices

    def set_proj_mobility(self):
        # TODO: implement mobility projections
        self.proj_mobility = {}

    def get_mobility_as_params(self):
        mobility_dict = self.actual_mobility
        mobility_dict.update(self.proj_mobility)
        tslices = list(mobility_dict.keys())
        params = {}
        if self.mobility_mode == "population_attached":
            matrix_list = [np.dot(mobility_dict[t]['dwell_rownorm'], np.transpose(mobility_dict[t]['dwell_colnorm'])) for t in tslices]
            for j, from_region in enumerate(self.regions):
                params[f"mob_{from_region}"] = [{'tslices': tslices[1:], 'attributes': {'region': to_region}, 'values': [m[i,j] for m in matrix_list]} for i, to_region in enumerate(self.regions)]
        elif self.mobility_mode == "location_attached":
            dwell_rownorm_list = [mobility_dict[t]['dwell_rownorm'] for t in tslices]
            dwell_colnorm_list = [mobility_dict[t]['dwell_colnorm'] for t in tslices]
            for j, in_region in enumerate(self.regions):
                params[f"mob_fracin_{in_region}"] = [{'tslices': tslices[1:], 'attributes': {'seir': 'S', 'region': to_region}, 'values': [m[i,j] for m in dwell_rownorm_list]} for i, to_region in enumerate(self.regions)]
            for i, from_region in enumerate(self.regions):
                for j, in_region in enumerate(self.regions):
                    params[f"mob_{in_region}_fracfrom_{from_region}"] = [{'tslices': tslices[1:], 'attributes': {},  'values': [m[i,j] for m in dwell_colnorm_list]}]
        else:
            ermsg = f'Mobility mode {self.mobility_mode} not supported'
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise ValueError(ermsg)
        # add in region populations as parameters for use later
        region_pops = {params_list['attributes']['region']: params_list['values'] for params_list in self.params_defs['total_pop'] }
        params.update(
            {f"region_pop_{region}": [{'tslices': None, 'attributes': {}, 'values': region_pops[region] }] for region in self.regions}
        )
        return params

    def get_kappas_as_params_using_region_fits(self, engine):
        # will set TC = 0.0 for the model and scale the "kappa" parameter for each region according to its fitted TC values so the forward sim can be run.
        # This approach does not currently support fitting, since only TC can be fit and there's only one set of TC for the model.
        results_list = []
        for i, tup in enumerate(zip(self.regions, self.region_fit_spec_ids)):
            region, spec_id = tup
            # load tslices and tcs from the database
            df = pd.read_sql_query(f"select regions, start_date, end_date, tslices, tc, from covid_model.specifications where spec_id = {spec_id}", con=engine, coerce_float=True)
            # make sure region in run spec matches our region
            if json.loads(df['regions'][0])[0] != region:
                ValueError(f'spec_id {spec_id} has region {json.loads(df["regions"][0])[0]} which does not match model\'s {i}th region: {region}')
            tslices = [(df['start_date'][0] + dt.timedelta(days=d).days - self.start_date).days for d in [0] + df['tslices'][0]]
            tc = df['tc'][0]
            results_list.append(pd.DataFrame.from_dict({'tslices': tslices, region: tc}).set_index('tslices'))
        # union all the region tslices
        df_tcs = pd.concat(results_list, axis=1)
        tslices = df_tcs.index.drop([0]).to_list()

        # retrieve prevalence data from db if allowing mobility
        prev_df = None
        if self.mobility_mode != 'none':
            results_list = []
            for region, spec_id, result_id in zip(self.regions, self.region_fit_spec_ids, self.region_fit_result_ids):
                df = pd.read_sql_query(f"SELECT t, vals FROM covid_model.results_v2 WHERE spec_id = {spec_id} AND result_id = {result_id} order by t", con=engine, coerce_float=True)
                df['infected'] = [sum(itemgetter('I', 'Ih')(json.loads(row))) for row in df['vals']]
                df['pop'] = [sum(json.loads(row).values()) for row in df['vals']]
                df = df.groupby('t').sum(['infected', 'pop'])
                df[region]=df['infected']/df['pop']
                df = df.drop(columns=['infected', 'pop'])
                results_list.append(df)
            prev_df = pd.concat(results_list, axis=1)

        # compute kappa parameter for each region and apply to model parameters
        params = {}
        if self.mobility_mode == 'none':
            self.tc_tslices = tslices
            self.tc = [0.0] * (len(tslices) + 1)
            params = {'kappa': [{'tslices': tslices, 'attributes': {'region': region}, 'values': [(1-tc) for tc in df_tcs[region]]} for region in df_tcs.columns]}
        elif self.mobility_mode == 'location_attached':
            # TODO: Implement
            params = {}
        elif self.mobility_mode == 'population_attached':
            # update tslices to include any t where mobility changes
            mob_tslices = np.array([t for t in self.actual_mobility.keys() if t >= self.tmin and t < self.tmax])
            prev_tslices = np.array([t for t in prev_df.index if t >= self.tmin and t < self.tmax])

            combined_tslices = [t.item() if isinstance(t, np.int32) else t for t in sorted(list(set([0] + tslices).union(set(mob_tslices)).union(set(prev_tslices))))]
            kappas = np.zeros(shape=[len(combined_tslices), len(self.regions)])
            for i, t in enumerate(combined_tslices):
                tc = df_tcs.iloc[df_tcs.index <= t,].iloc[-1].to_numpy()
                prev = prev_df.iloc[prev_tslices[prev_tslices <= t][-1]].to_numpy()
                mobs = self.actual_mobility[mob_tslices[mob_tslices <= t][-1]]
                kappas[i,] = (1-tc) * prev / np.linalg.multi_dot([mobs['dwell_rownorm'], np.transpose(mobs['dwell_colnorm']), np.transpose(prev)])
            np.nan_to_num(kappas, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            self.tc_tslices = combined_tslices[1:]
            self.tc = [0.0] * len(combined_tslices)
            params = {'kappa_pa': [{'tslices': combined_tslices[1:], 'attributes': {'region': region}, 'values': kappas[:,j].tolist()} for j, region in enumerate(self.regions)]}
        return params

    # pulls hospitalizations for only the first region, since the model only fits one region at a time
    def set_actual_hosp(self, engine):
        logger.info(f"{str(self.tags)} Retrieving hospitalizations")
        # makes sure we pull from EMResource if region is CO
        county_ids = self.region_defs[self.regions[0]]['counties_fips'] if self.regions[0] != 'co' else None
        hosps = ExternalHosps(engine).fetch(county_ids=county_ids)['currently_hospitalized']
        # fill in the beginning if necessary
        if min(hosps.index.get_level_values(0)) > self.start_date:
            s_fill = pd.Series(index=[self.t_to_date(t) for t in range(self.tmin, self.date_to_t(min(hosps.index.get_level_values(0))))], dtype='float64').fillna(0)
            self.actual_hosp = pd.concat([s_fill, hosps])
        else:
            self.actual_hosp = hosps[hosps.index.get_level_values(0) >= self.start_date]

    ####################################################################################################################
    ### Properites

    ### Date / Time. Updating start date and end date updates other date/time attributes also
    @property
    def start_date(self):
        return self.__start_date

    @start_date.setter
    def start_date(self, value):
        self.__start_date = value if isinstance(value, dt.date) else dt.datetime.strptime(value, "%Y-%m-%d").date()
        self.__tmax = (self.end_date - self.start_date).days
        self.__trange = range(self.tmin, self.tmax + 1)
        self.__daterange = pd.date_range(self.start_date, end=self.end_date).date

    @property
    def end_date(self):
        return self.__end_date

    @end_date.setter
    def end_date(self, value):
        self.__end_date = value if isinstance(value, dt.date) else dt.datetime.strptime(value, "%Y-%m-%d").date()
        self.__tmax = (self.end_date - self.start_date).days
        self.__trange = range(self.tmin, self.tmax + 1)
        self.__daterange =  pd.date_range(self.start_date, end=self.end_date).date

    @property
    def tmin(self):
        return 0

    @property
    def tmax(self):
        return self.__tmax

    @tmax.setter
    def tmax(self, value):
        self.__tmax = value
        self.__end_date = self.start_date + dt.timedelta(days=value)
        self.__trange = range(self.tmin, self.tmax + 1)
        self.__daterange = pd.date_range(self.start_date, end=self.end_date).date

    @property
    def trange(self):
        return self.__trange

    @property
    def daterange(self):
        return self.__daterange

    @property
    def attr_names(self):
        return list(self.attrs.keys())

    @property
    def param_attr_names(self):
        return self.attr_names[1:]

    ### regions
    @property
    def regions(self):
        return self.__regions

    @regions.setter
    def regions(self, value: list):
        self.__regions = value
        self.attrs['region'] = value  # if regions changes, update the compartment attributes also


    ### things which are dictionaries but which may be given as a path to a json file
    @property
    def params_defs(self):
        return self.__params_defs

    @params_defs.setter
    def params_defs(self, value):
        self.__params_defs = value if isinstance(value, dict) else json.load(open(value))

    @property
    def vacc_proj_params(self):
        return self.__vacc_proj_params

    @vacc_proj_params.setter
    def vacc_proj_params(self, value):
        self.__vacc_proj_params = value if isinstance(value, dict) else json.load(open(value))

    @property
    def region_defs(self):
        return self.__region_defs

    @region_defs.setter
    def region_defs(self, value):
        self.__region_defs = value if isinstance(value, dict) else json.load(open(value))

    ### Set mobility mode as None if "none"
    @property
    def mobility_mode(self):
        return self.__mobility_mode

    @mobility_mode.setter
    def mobility_mode(self, value):
        self.__mobility_mode = value if value != 'none' else None


    ### Properties that take a little computation to get

    # initial state y0, expressed as a dictionary with non-empty compartments as keys
    @property
    def y0_dict(self):
        group_pops = {(region, age): self.params_by_t['all'][f'{region}_{age_param}_pop'][0] for region in self.regions for age, age_param in zip(['0-19', '20-39', '40-64', '65+'], ['0_19', '20_39', '40_64', '65p'])}
        y0d = {('S', age, 'none', 'none', 'none', 'none', region): n for (region, age), n in group_pops.items()}
        return y0d

    # return the parameters as nested dictionaries
    @property
    def params_as_dict(self, params=None):
        params_dict = {}
        for cmpt, cmpt_dict in self.params_by_t.items():
            key = f"({','.join(cmpt)})" if isinstance(cmpt, tuple) else cmpt
            params_dict[key] = cmpt_dict
        return params_dict

    @property
    def n_compartments(self):
        return len(self.cmpt_idx_lookup)

    @property
    def solution_y(self):
        return np.transpose(self.solution.y)

    @property
    def solution_ydf(self):
        return pd.concat([self.y_to_series(self.solution_y[t]) for t in self.trange], axis=1, keys=self.trange, names=['t']).transpose()


    ####################################################################################################################
    ### useful getters

    def date_to_t(self, date):
        if isinstance(date, str):
            return (dt.datetime.strptime(date, "%Y-%m-%d").date() - self.start_date).days
        else:
            return (date - self.start_date).days

    def t_to_date(self, t):
        return self.start_date + dt.timedelta(days=t)

    def get_vacc_rates(self):
        df = pd.concat([self.actual_vacc_df, self.proj_vacc_df])
        return df

    # create a y0 vector with all values as 0, except those designated in y0_dict
    def y0_from_dict(self, y0_dict):
        y0 = [0] * self.n_compartments
        for cmpt, n in y0_dict.items():
            y0[self.cmpt_idx_lookup[cmpt]] = n
        return y0

    # returns list of fips codes for each county in the given region, or every region in this model if not given
    def get_all_county_fips(self, regions=None):
        regions = self.regions if regions is None else regions
        return [county_fips for region in regions for county_fips in self.region_defs[region]['counties_fips']]

    # convert y-array to series with compartment attributes as multiindex
    def y_to_series(self, y):
        return pd.Series(index=self.compartments_as_index, data=y)

    # give the counts for all compartments over time, but group by the compartment attributes listed
    def solution_sum(self, group_by_attr_levels):
        df = self.solution_ydf.groupby(group_by_attr_levels, axis=1).sum()
        df['date'] = self.daterange
        df = df.set_index('date')
        return df

    def solution_sum_Ih(self):
        return pd.Series(self.solution_y[:, self.compartments_as_index.get_level_values(0) == "Ih"].sum(axis=1), index=self.daterange)

    # Get the immunity against a given variant
    def immunity(self, variant='omicron', vacc_only=False, to_hosp=False, age=None):
        params = self.params_as_df
        group_by_attr_names = [attr_name for attr_name in self.param_attr_names if attr_name != 'variant']
        n = self.solution_sum(group_by_attr_names).stack(level=group_by_attr_names)

        if age is not None:
            params = params.xs(age, level='age')
            n = n.xs(age, level='age')

        if vacc_only:
            params.loc[params.index.get_level_values('vacc') == 'none', 'immunity'] = 0
            params.loc[params.index.get_level_values('vacc') == 'none', 'severe_immunity'] = 0

        variant_params = params.xs(variant, level='variant')
        if to_hosp:
            weights = variant_params['hosp'] * n
            return (weights * (
                        1 - (1 - variant_params['immunity']) * (1 - variant_params['severe_immunity']))).groupby(
                't').sum() / weights.groupby('t').sum()
        else:
            return (n * variant_params['immunity']).groupby('t').sum() / n.groupby('t').sum()

    def modeled_vs_actual_hosps(self):
        df = self.solution_sum(['seir', 'region'])['Ih'].stack('region').rename('modeled').to_frame()
        df['actual'] = self.actual_hosp[:len(self.daterange)].to_numpy()
        df = df.reindex(columns=['actual', 'modeled'])
        return df

    # get a parameter for a given set of attributes and trange
    def get_param(self, param, attrs=None, trange=None):
        # pass empty dict if want all compartments listed separately
        if trange is None:
            trange = self.trange
        cmpt_list = ['all'] if attrs is None else self.filter_cmpts_by_attrs(attrs, is_param_cmpts=True) if attrs else self.param_compartments
        vals = []
        for cmpt in cmpt_list:
            df = pd.DataFrame(index=trange, columns=[param])
            df.index.names = ['t']
            if param in self.params_by_t[cmpt].keys():
                for t in trange:
                    t_left = self.params_by_t[cmpt][param].keys()[self.params_by_t[cmpt][param].bisect(t) - 1]
                    df.loc[t][param] = self.params_by_t[cmpt][param][t_left]
                vals.append((cmpt, df))
        return vals

    # get all terms that refer to flow from one specific compartment to another
    def get_terms_by_cmpt(self, from_cmpt, to_cmpt):
        return [term for term in self.terms if term.from_cmpt_idx == self.cmpt_idx_lookup[from_cmpt] and term.to_cmpt_idx == self.cmpt_idx_lookup[to_cmpt]]

    # get the terms that refer to flow from compartments with a set of attributes to compartments with another set of attributes
    def get_terms_by_attr(self, from_attrs, to_attrs):
        idx = [i for i, term in enumerate(self.terms) if self.does_cmpt_have_attrs(self.compartments[term.from_cmpt_idx], from_attrs) and self.does_cmpt_have_attrs(self.compartments[term.to_cmpt_idx], to_attrs)]
        return [self.terms[i] for i in idx]

    # create a json string capturing all of the ode terms: nonlinear, linear, and constant
    def ode_terms_as_json(self, compact=False):
        if compact:
            cm = ", ".join([f'[{i},{c}]' for i, c in enumerate(self.compartments)])
            cv = [[t, spsp.csr_array(vec)] for t, vec in self.constant_vector.items() if any(vec != 0)]
            cv = {t: ' ,'.join([f'({idx},{val:.2e})' for idx, val in zip(m.nonzero()[1].tolist(), m[m.nonzero()].tolist())]) for t, m in cv}
            lm = {t: ' ,'.join([f'({idx1},{idx2},{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[0].tolist(), m.nonzero()[1].tolist(), m[m.nonzero()].A[0].tolist())]) for t, m in self.linear_matrix.items() if len(m.nonzero()[0]) > 0}
            nl = {t: {f'({",".join([f"{k}" for k in keys])})': ', '.join([f'({idx1},{idx2},{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[0].tolist(), m.nonzero()[1].tolist(), m[m.nonzero()].A[0].tolist()) if val != 0]) for keys, m in mat_dict.items()} for t, mat_dict in self.nonlinear_matrices.items() if len(mat_dict) > 0}
            nlm = self.nonlinear_multiplier
            return json.dumps({"compartments": cm, "constant_vector": cv, "linear_matrix": lm, "nonlinear_multiplier": nlm, "nonlinear_matrices": nl}, indent=2)
        else:
            def fcm(i):
                return f'{",".join(self.compartments[i])}'

            cv = [[t, spsp.csr_array(vec)] for t, vec in self.constant_vector.items() if any(vec != 0)]
            cv = {t: {fcm(idx): f'{val:.2e}' for idx, val in zip(m.nonzero()[1].tolist(), m[m.nonzero()].tolist())} for t, m in cv}
            lm = {t: {f'({fcm(idx1)};{fcm(idx2)}': f'{val:.2e}' for idx1, idx2, val in zip(m.nonzero()[1].tolist(), m.nonzero()[0].tolist(), m[m.nonzero()].A[0].tolist())} for t, m in self.linear_matrix.items() if len(m.nonzero()[0]) > 0}
            nl = {t: {f'({";".join([f"{fcm(k)}" for k in keys])})': {f'({fcm(idx1)};{fcm(idx2)})': f'{val:.2e})' for idx1, idx2, val in zip(m.nonzero()[1].tolist(), m.nonzero()[0].tolist(), m[m.nonzero()].A[ 0].tolist()) if val != 0} for keys, m in mat_dict.items()} for t, mat_dict in self.nonlinear_matrices.items() if len(mat_dict) > 0}
            nlm = self.nonlinear_multiplier
            return json.dumps({"constant_vector": cv, "linear_matrix": lm, "nonlinear_multiplier": nlm, "nonlinear_matrices": nl}, indent=2)



    ####################################################################################################################
    ### ODE related functions

    # check if a cmpt matches a dictionary of attributes
    def does_cmpt_have_attrs(self, cmpt, attrs, is_param_cmpts=False):
        return all(
            cmpt[self.param_attr_names.index(attr_name) if is_param_cmpts else list(self.attrs.keys()).index(attr_name)]
            in ([attr_val] if isinstance(attr_val, str) else attr_val)
            for attr_name, attr_val in attrs.items())

    # return compartments that match a dictionary of attributes
    def filter_cmpts_by_attrs(self, attrs, is_param_cmpts=False):
        return [cmpt for cmpt in (self.param_compartments if is_param_cmpts else self.compartments) if self.does_cmpt_have_attrs(cmpt, attrs, is_param_cmpts)]


    ####################################################################################################################
    ### Prepping and Running

    def get_vacc_per_available(self):
        vacc_rates = self.get_vacc_rates()
        missing_dates = pd.DataFrame({'date': [d for d in self.daterange if d < min(vacc_rates.index.get_level_values('date'))]})
        missing_shots = missing_dates.merge(pd.DataFrame(index=vacc_rates.reset_index('date').index.unique(), columns=vacc_rates.columns).fillna(0).reset_index(), 'cross').set_index(['date', 'region', 'age'])
        vacc_rates = pd.concat([missing_shots, vacc_rates])
        vacc_rates['t'] = [self.date_to_t(d) for d in vacc_rates.index.get_level_values('date')]
        vacc_rates = vacc_rates.set_index('t', append=True)
        populations = pd.concat([self.get_param(f'{region}_{grp}_pop', trange=vacc_rates.index.get_level_values('t').unique())[0][1].rename(columns={f'{region}_{grp}_pop':'population'}).assign(age=grp_lab, region=region) for grp, grp_lab in zip(['0_19', '20_39', '40_64', '65p'], ['0-19', '20-39', '40-64', '65+']) for region in self.regions])
        populations = populations.reset_index().set_index(['region', 'age', 't'])
        cumu_vacc = vacc_rates.groupby(['region', 'age']).cumsum()
        cumu_vacc_final_shot = cumu_vacc - cumu_vacc.shift(-1, axis=1).fillna(0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.join(populations)
        # vaccinations eventually overtake population (data issue) which would make 'none' < 0 so clip at 0
        cumu_vacc_final_shot['none'] = (cumu_vacc_final_shot['population'] * 2 - cumu_vacc_final_shot.sum(axis=1)).clip(lower=0)
        cumu_vacc_final_shot = cumu_vacc_final_shot.drop(columns='population')
        cumu_vacc_final_shot = cumu_vacc_final_shot.reindex(columns=['none', 'shot1', 'shot2', 'shot3'])

        available_for_vacc = cumu_vacc_final_shot.shift(1, axis=1).drop(columns='none')
        vacc_per_available = (vacc_rates / available_for_vacc).fillna(0).replace(np.inf, 0).reorder_levels(['t', 'date', 'region', 'age']).sort_index()
        # because vaccinations exceed the population, we can get rates greater than 1. To prevent compartments have negative people, we have to cap the rate at 1
        vacc_per_available = vacc_per_available.clip(upper=1)
        return vacc_per_available

    # set values for a single parameter based on param_tslices
    def set_param(self, param, attrs: dict=None, vals: dict=None, mults: dict=None, desc=None):
        # get only the compartments we want
        cmpts = ['all'] if attrs is None else self.filter_cmpts_by_attrs(attrs, is_param_cmpts=True)
        # update the parameter
        for cmpt in cmpts:
            if param not in self.params_by_t[cmpt].keys():
                # take from global params if present, otherwise create new dictionary
                if param in self.params_by_t['all'].keys():
                    self.params_by_t[cmpt][param] = copy.deepcopy(self.params_by_t['all'][param])
                else:
                    self.params_by_t[cmpt][param] = SortedDict()
            if vals is not None:
                for d, val in vals.items():
                    t = max(self.date_to_t(d), self.tmin)
                    if t > self.tmax:
                        continue
                    self.params_by_t[cmpt][param][t] = val
            if mults is not None:
                # add in tslices which are missing
                for d in sorted(list(mults.keys())):
                    t = max(self.date_to_t(d), self.tmin)
                    if t > self.tmax:
                        continue
                    if not self.params_by_t[cmpt][param].__contains__(t):
                        t_left = self.params_by_t[cmpt][param].keys()[self.params_by_t[cmpt][param].bisect(t) - 1]
                        self.params_by_t[cmpt][param][t] = self.params_by_t[cmpt][param][t_left]
                # set val or multiply
                for d, mult in mults.items():
                    t = max(self.date_to_t(d), self.tmin)
                    if t > self.tmax:
                        continue
                    self.params_by_t[cmpt][param][t] *= mult

    # combine param_defs, vaccine_defs, etc. into a time indexed parameters dictionary
    def build_param_lookups(self, apply_vaccines=True, vacc_delay=14):
        logger.debug(f"{str(self.tags)} Building param lookups")
        self.compartments_as_index = pd.MultiIndex.from_product(self.attrs.values(), names=self.attrs.keys())
        self.compartments = list(self.compartments_as_index)
        self.cmpt_idx_lookup = pd.Series(index=self.compartments_as_index, data=range(len(self.compartments_as_index))).to_dict()
        self.param_compartments = list(set(tuple(attr_val for attr_val, attr_name in zip(cmpt, self.attr_names) if attr_name in self.param_attr_names) for cmpt in self.compartments))
        self.params_by_t = {pcmpt: {} for pcmpt in ['all'] + self.param_compartments}

        for param_def in self.params_defs:
            self.set_param(**param_def)

        # determine global trange
        self.params_trange = sorted(list(set.union(*[set(cmpt_param.keys()) for cmpt in self.params_by_t.values() for cmpt_param in cmpt.values()])))
        self.t_prev_lookup = {t_int: max(t for t in self.params_trange if t <= t_int) for t_int in self.trange}

        if apply_vaccines:
            vacc_per_available = self.get_vacc_per_available()

            # apply vacc_delay
            vacc_per_available = vacc_per_available.groupby(['region', 'age']).shift(vacc_delay).fillna(0)

            # group vacc_per_available by trange interval
            t_index_rounded_down_to_tslices = pd.cut(vacc_per_available.index.get_level_values('t'), self.params_trange + [self.tmax], right=False, retbins=False, labels=self.params_trange)
            vacc_per_available = vacc_per_available.groupby([t_index_rounded_down_to_tslices, 'region', 'age']).mean()
            vacc_per_available['date'] = [self.t_to_date(d) for d in vacc_per_available.index.get_level_values(0)]
            vacc_per_available = vacc_per_available.reset_index().set_index(['region', 'age']).sort_index()

            # set the fail rate and vacc per unvacc rate for each dose
            for shot in self.attrs['vacc'][1:]:
                for age in self.attrs['age']:
                    for region in self.attrs['region']:
                        vpa_sub = vacc_per_available[['date', shot]].loc[region, age]
                        vpa_sub = vpa_sub.reset_index(drop=True).set_index('date').sort_index().drop_duplicates().to_dict()[shot]
                        self.set_param(param=f'{shot}_per_available', attrs={'age': age, 'region': region}, vals=vpa_sub)

    # set TC by slice, and update non-linear multiplier; defaults to reseting the last TC values
    def apply_tc(self, tcs=None, tslices=None, force_nlm_update=False):
        # if tslices are provided, replace any tslices >= tslices[0] with the new tslices
        if tslices is not None:
            self.tc_tslices = [tslice for tslice in self.tc_tslices if tslice < tslices[0]] + tslices
            self.tc = self.tc[:len(self.tc_tslices) + 1]  # truncate tc if longer than tslices
            self.tc += [self.tc[-1] if len(self.tc)>0 else 0] * (1 + len(self.tc_tslices) - len(self.tc))  # extend tc if shorter than tslices
            self.tc_t_prev_lookup = {t_int: max(t for t in [0] + self.tc_tslices if t <= t_int) for t_int in self.trange}

        # if tc is provided, replace the end bit
        if tcs is not None:
            self.tc = self.tc[:-len(tcs)] + tcs

        # if the lengths do not match, raise an error
        if tcs is not None and tslices is not None and len(self.tc) != len(self.tc_tslices) + 1:
            ermsg = f'The length of tc ({len(self.tc)}) must be equal to the length of tc_tslices ({len(self.tc_tslices)}) + 1.'
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise ValueError(ermsg)

        # apply the new TC values to the non-linear multiplier to update the ODE
        # only update the nonlinear multipliers for TCs that have been changed
        if tcs is not None:
            for tslice, tc in zip(([0] + self.tc_tslices)[-len(self.tc):], tcs):
                self.nonlinear_multiplier[self.tc_t_prev_lookup[tslice]] = 1-tc
        # update all multipliers using stored tc's if requested
        if force_nlm_update:
            for tslice, tc in zip([0] + self.tc_tslices, self.tc):
                self.nonlinear_multiplier[self.tc_t_prev_lookup[tslice]] = 1-tc

    def _default_nonlinear_matrix(self):
        return spsp.lil_matrix((self.n_compartments, self.n_compartments))

    # assign default values to matrices
    def reset_ode(self):
        self.terms = []
        self.linear_matrix = {t: spsp.lil_matrix((self.n_compartments, self.n_compartments)) for t in self.params_trange}
        self.nonlinear_matrices = {t: defaultdict(self._default_nonlinear_matrix) for t in self.params_trange}
        self.constant_vector = {t: np.zeros(self.n_compartments) for t in self.params_trange}
        self.nonlinear_multiplier = {}

     # takes a symbolic expression (coef), and looks up variable names in params to provide a computed output for each t in trange
    def calc_coef_by_t(self, coef, cmpt):
        if len(cmpt) > len(self.param_attr_names):
            param_cmpt = tuple(attr for attr, level in zip(cmpt, self.attr_names) if level in self.param_attr_names)
        else:
            param_cmpt = cmpt

        if isinstance(coef, dict):
            return {t: coef[t] if t in coef.keys() else 0 for t in self.params_trange}
        elif callable(coef):
            return {t: coef(t) for t in self.params_trange}
        elif isinstance(coef, str):
            if coef == '1':
                coef_by_t = {t: 1 for t in self.params_trange}
            else:
                coef_by_t = {}
                expr = parse_expr(coef)
                relevant_params = [str(s) for s in expr.free_symbols]
                if len(relevant_params) == 1 and coef == relevant_params[0]:
                    actual_param_cmpt = param_cmpt if coef in self.params_by_t[param_cmpt].keys() else 'all'
                    param_vals = {t: self.params_by_t[actual_param_cmpt][coef][t] if self.params_by_t[actual_param_cmpt][coef].__contains__(t) else None for t in self.params_trange}
                    for i, t in enumerate(self.params_trange[1:]):
                        if param_vals[t] is None:
                            param_vals[t] = param_vals[self.params_trange[i]]
                    coef_by_t = param_vals
                else:
                    func = sym.lambdify(relevant_params, expr)
                    param_vals = {t: {} for t in self.params_trange}
                    for param in relevant_params:
                        actual_param_cmpt = param_cmpt if param in self.params_by_t[param_cmpt].keys() else 'all'
                        for t in self.params_trange:
                            param_vals[t][param] = self.params_by_t[actual_param_cmpt][param][t] if self.params_by_t[actual_param_cmpt][param].__contains__(t) else None
                        for i, t in enumerate(self.params_trange[1:]):
                            if param_vals[t][param] is None:
                                param_vals[t][param] = param_vals[self.params_trange[i]][param]
                    for t, tvals in param_vals.items():
                        coef_by_t[t] = func(**tvals)
            return coef_by_t
        else:
            return {t: coef for t in self.params_trange}

    # add a flow term, and add new flow to ODE matrices
    def add_flow_from_cmpt_to_cmpt(self, from_cmpt, to_cmpt, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
        if len(from_cmpt) < len(self.attrs.keys()):
            ermsg = f'The length of tc ({len(self.tc)}) must be equal to the length of tc_tslices ({len(self.tc_tslices)}) + 1.'
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise ValueError(ermsg)
        if len(to_cmpt) < len(self.attrs.keys()):
            ermsg = f'Destination compartment `{to_cmpt}` does not have the right number of attributes.'
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise ValueError(ermsg)
        if scale_by_cmpts is not None:
            for cmpt in scale_by_cmpts:
                if len(cmpt) < len(self.attrs.keys()):
                    ermsg = f'Scaling compartment `{cmpt}` does not have the right number of attributes.'
                    logger.exception(f"{str(self.tags)}" + ermsg)
                    raise ValueError(ermsg)

        if coef is not None:
            if scale_by_cmpts_coef:
                coef_by_t_lookup = {c: self.calc_coef_by_t(c, to_cmpt) for c in set(scale_by_cmpts_coef)}
                coef_by_t_ld = [coef_by_t_lookup[c] for c in scale_by_cmpts_coef]
                coef_by_t_dl = {t: [dic[t] for dic in coef_by_t_ld] for t in self.params_trange}
            else:
                coef_by_t_dl = None

        term = ODEFlowTerm.build(
            from_cmpt_idx=self.cmpt_idx_lookup[from_cmpt],
            to_cmpt_idx=self.cmpt_idx_lookup[to_cmpt],
            coef_by_t=self.calc_coef_by_t(coef, to_cmpt),  # switched BACK to setting parameters use the TO cmpt
            scale_by_cmpts_idxs=[self.cmpt_idx_lookup[cmpt] for cmpt in
                                 scale_by_cmpts] if scale_by_cmpts is not None else None,
            scale_by_cmpts_coef_by_t=coef_by_t_dl if scale_by_cmpts is not None else None,
            constant_by_t=self.calc_coef_by_t(constant, to_cmpt) if constant is not None else None)

        if not (isinstance(term, ConstantODEFlowTerm)) and all([c == 0 for c in term.coef_by_t.values()]):
            pass
        elif isinstance(term, ConstantODEFlowTerm) and all([c == 0 for c in term.constant_by_t.values()]):
            pass
        else:
            self.terms.append(term)

            # add term to matrices
            for t in self.params_trange:
                term.add_to_linear_matrix(self.linear_matrix[t], t)
                term.add_to_nonlinear_matrices(self.nonlinear_matrices[t], t)
                term.add_to_constant_vector(self.constant_vector[t], t)

    # add multipler flows, from all compartments with from_attrs, to compartments that match the from compartments, but replacing attributes as designated in to_attrs
    # e.g. from {'seir': 'S', 'age': '0-19'} to {'seir': 'E'} will be a flow from susceptible 0-19-year-olds to exposed 0-19-year-olds
    def add_flows_from_attrs_to_attrs(self, from_attrs, to_attrs, coef=None, scale_by_cmpts=None, scale_by_cmpts_coef=None, constant=None):
        from_cmpts = self.filter_cmpts_by_attrs(from_attrs)
        for from_cmpt in from_cmpts:
            to_cmpt_list = list(from_cmpt)
            for attr_name, new_attr_val in to_attrs.items():
                to_cmpt_list[list(self.attrs.keys()).index(attr_name)] = new_attr_val
            to_cmpt = tuple(to_cmpt_list)
            self.add_flow_from_cmpt_to_cmpt(from_cmpt, to_cmpt, coef=coef, scale_by_cmpts=scale_by_cmpts,
                                            scale_by_cmpts_coef=scale_by_cmpts_coef, constant=constant)

    # build ODE
    def build_ode_flows(self):
        logger.debug(f"{str(self.tags)} Building ode flows")
        self.reset_ode()
        self.apply_tc(force_nlm_update=True)  # update the nonlinear multiplier

        # vaccination
        self.add_flows_from_attrs_to_attrs({'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'weak'}, coef=f'shot1_per_available * (1 - shot1_fail_rate)')
        self.add_flows_from_attrs_to_attrs({'vacc': f'none'}, {'vacc': f'shot1', 'immun': f'none'}, coef=f'shot1_per_available * shot1_fail_rate')
        for i in [2, 3]:
            for immun in self.attrs['immun']:
                # if immun is none, that means that the first vacc shot failed, which means that future shots may fail as well
                if immun == 'none':
                    self.add_flows_from_attrs_to_attrs({'vacc': f'shot{i - 1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, coef=f'shot{i}_per_available * (1 - shot{i}_fail_rate / shot{i - 1}_fail_rate)')
                    self.add_flows_from_attrs_to_attrs({'vacc': f'shot{i - 1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'none'}, coef=f'shot{i}_per_available * (shot{i}_fail_rate / shot{i - 1}_fail_rate)')
                else:
                    self.add_flows_from_attrs_to_attrs({'vacc': f'shot{i - 1}', "immun": immun}, {'vacc': f'shot{i}', 'immun': f'strong'}, coef=f'shot{i}_per_available')

        # seed variants
        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'none'}, constant='initial_seed')
        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'alpha'}, constant='alpha_seed')
        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'delta'}, constant='delta_seed')
        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'omicron'}, constant='om_seed')
        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'age': '40-64', 'vacc': 'none', 'variant': 'none', 'immun': 'none'}, {'seir': 'E', 'variant': 'ba2'}, constant='ba2_seed')

        # exposure
        for variant in self.attrs['variant']:
            # No mobility between regions (or a single region)
            if self.mobility_mode is None or self.mobility_mode == "none":
                for region in self.attrs['region']:
                    asymptomatic_transmission = f'(1 - immunity) * kappa * betta / {region}_pop'
                    sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant, 'region': region})
                    asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant, 'region': region})
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'variant': 'none', 'region': region}, {'seir': 'E', 'variant': variant}, coef=f'lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
                    self.add_flows_from_attrs_to_attrs({'seir': 'S', 'variant': 'none', 'region': region}, {'seir': 'E', 'variant': variant}, coef=asymptomatic_transmission, scale_by_cmpts=asympt_cmpts)
            # Transmission parameters attached to the susceptible population
            elif self.mobility_mode == "population_attached":
                for from_region in self.attrs['region']:
                    # kappa in this mobility mode is associated with the susceptible population, so no need to store every kappa in every region
                    asymptomatic_transmission = f'(1 - immunity) * kappa_pa * betta / {from_region}_pop'
                    for to_region in self.attrs['region']:
                        sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant, 'region': from_region})
                        asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant, 'region': from_region})
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_{from_region} * lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
                        self.add_flows_from_attrs_to_attrs({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_{from_region} * {asymptomatic_transmission}', scale_by_cmpts=asympt_cmpts)
            # Transmission parameters attached to the transmission location
            elif self.mobility_mode == "location_attached":
                for from_region in self.attrs['region']:
                    for in_region in self.attrs['region']:
                        # kappa in this mobility mode is associated with the in_region, so need to store every kappa in every region
                        asymptomatic_transmission = f'(1 - immunity) * kappa_la_{in_region} * betta / {from_region}_pop'
                        for to_region in self.attrs['region']:
                            sympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'I', 'variant': variant, 'region': from_region})
                            asympt_cmpts = self.filter_cmpts_by_attrs({'seir': 'A', 'variant': variant, 'region': from_region})
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_fracin_{in_region} * mob_{in_region}_fracfrom_{from_region} * lamb * {asymptomatic_transmission}', scale_by_cmpts=sympt_cmpts)
                            self.add_flows_from_attrs_to_attrs({'seir': 'S', 'variant': 'none', 'region': to_region}, {'seir': 'E', 'variant': variant}, coef=f'mob_fracin_{in_region} * mob_{in_region}_fracfrom_{from_region} * {asymptomatic_transmission}', scale_by_cmpts=asympt_cmpts)

        # disease progression
        self.add_flows_from_attrs_to_attrs({'seir': 'E'}, {'seir': 'I'}, coef='1 / alpha * pS')
        self.add_flows_from_attrs_to_attrs({'seir': 'E'}, {'seir': 'A'}, coef='1 / alpha * (1 - pS)')
        # assume noone is receiving both pax and mab
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp * (1 - severe_immunity) * ((1-mab_prev) + (1-pax_prev))')
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp * (1 - severe_immunity) * mab_prev * mab_hosp_adj')
        self.add_flows_from_attrs_to_attrs({'seir': 'I'}, {'seir': 'Ih'}, coef='gamm * hosp * (1 - severe_immunity) * pax_prev * pax_hosp_adj')

        # disease termination
        for variant in self.attrs['variant']:
            priorinf = variant if variant != 'none' and variant in self.attrs['priorinf'] else 'non-omicron'
            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='gamm * (1 - hosp - dnh) * (1 - priorinf_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * (1 - hosp - dnh) * priorinf_fail_rate')
            self.add_flows_from_attrs_to_attrs({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='gamm * (1 - priorinf_fail_rate)')
            self.add_flows_from_attrs_to_attrs({'seir': 'A', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * priorinf_fail_rate')

            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='1 / hlos * (1 - dh) * (1 - priorinf_fail_rate) * (1-mab_prev)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * (1 - dh) * priorinf_fail_rate * (1-mab_prev)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf, 'immun': 'strong'}, coef='1 / (hlos * mab_hlos_adj) * (1 - dh) * (1 - priorinf_fail_rate) * mab_prev')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'S', 'variant': 'none', 'priorinf': priorinf}, coef='1 / (hlos * mab_hlos_adj) * (1 - dh) * priorinf_fail_rate * mab_prev')

            self.add_flows_from_attrs_to_attrs({'seir': 'I', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='gamm * dnh * (1 - severe_immunity)')
            self.add_flows_from_attrs_to_attrs({'seir': 'Ih', 'variant': variant}, {'seir': 'D', 'variant': 'none', 'priorinf': priorinf}, coef='1 / hlos * dh')

        # immunity decay
        self.add_flows_from_attrs_to_attrs({'immun': 'strong'}, {'immun': 'weak'}, coef='1 / imm_decay_days')

    # convert ODE matrices to CSR format, to (massively) improve performance
    def compile(self):
        logger.debug(f"{str(self.tags)} compiling ODE")
        for t in self.params_trange:
            self.linear_matrix[t] = self.linear_matrix[t].tocsr()
            for k, v in self.nonlinear_matrices[t].items():
                self.nonlinear_matrices[t][k] = v.tocsr()

    # ODE step forward
    def ode(self, t: float, y: list):
        dy = [0] * self.n_compartments
        t_int = self.t_prev_lookup[math.floor(t)]
        t_nlm = self.tc_t_prev_lookup[math.floor(t)]

        # apply linear terms
        dy += (self.linear_matrix[t_int]).dot(y)

        # apply non-linear terms
        for scale_by_cmpt_idxs, matrix in self.nonlinear_matrices[t_int].items():
            dy += self.nonlinear_multiplier[t_nlm] * sum(itemgetter(*scale_by_cmpt_idxs)(y)) * (matrix).dot(y)

        # apply constant terms
        dy += self.constant_vector[t_int]

        return dy

    # solve ODE using scipy.solve_ivp, and put solution in solution_y and solution_ydf
    # TODO: try Julia ODE package, to improve performance
    def solve_seir(self, y0=None, method='RK45'):
        if y0 is None:
            y0 = self.y0_from_dict(self.y0_dict)
        self.solution = spi.solve_ivp(
            fun=self.ode,
            t_span=[min(self.trange), max(self.trange)],
            y0=y0,
            t_eval=self.trange,
            method=method,
            max_step=self.max_step_size
        )
        if not self.solution.success:
            ermsg = f'ODE solver failed with message: {self.solution.message}'
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise RuntimeError(ermsg)

    # a model must be prepped before it can be run; if any params EXCEPT the TC change, it must be re-prepped
    def prep(self, rebuild_param_lookups=True, **build_param_lookup_args):
        logger.info(f"{str(self.tags)} Prepping Model")
        if rebuild_param_lookups:
            self.build_param_lookups(**build_param_lookup_args)
        self.build_ode_flows()
        self.compile()

    ####################################################################################################################
    ### Reading and Writing Data

    def serialize_vacc(self, df):
        df = df.reset_index()
        df['date'] = [dt.datetime.strftime(d, "%Y-%m-%d") for d in df['date']]
        return df.to_dict('records')

    @classmethod
    def unserialize_vacc(cls, vdict):
        df = pd.DataFrame.from_dict(vdict)
        df['date'] = [dt.datetime.strptime(d, "%Y-%m-%d") for d in df['date']]
        return df.set_index(['date', 'region', 'age'])

    def serialize_hosp(self, df):
        df = pd.DataFrame({'date': df.index, 'hosps': df.values})
        df['date'] = [dt.datetime.strftime(d, "%Y-%m-%d") for d in df['date']]
        return df.to_dict('records')

    @classmethod
    def unserialize_hosp(cls, hdict):
        df = pd.DataFrame.from_dict(hdict)
        df['date'] = [dt.datetime.strptime(d, "%Y-%m-%d") for d in df['date']]
        return df.set_index(['date'])['hosps']

    # serializes SOME of this model's properties to a json format which can be written to database
    # model needs prepping still
    def to_json_string(self):
        logger.debug(f"{str(self.tags)} Serializing model to json")
        keys = ['base_spec_id', 'spec_id', 'region_fit_spec_ids', 'region_fit_result_ids', 'tags', '_CovidModel__start_date', '_CovidModel__end_date', 'attrs', 'tc_tslices', 'tc', 'tc_cov', 'tc_tslices', 'tc_t_prev_lookup', '_CovidModel__params_defs',
                '_CovidModel__region_defs', '_CovidModel__regions', '_CovidModel__vacc_proj_params', '_CovidModel__mobility_mode', 'actual_mobility', 'mobility_proj_params', 'actual_vacc_df', 'proj_vacc_df', 'actual_hosp']
        #TODO: handle mobility, add in proj_mobility
        serial_dict = OrderedDict()
        for key in keys:
            val = self.__dict__[key]
            if isinstance(val, dt.date):
                serial_dict[key] = val.strftime('%Y-%m-%d')
            elif isinstance(val, np.ndarray):
                serial_dict[key] = val.tolist()
            elif key in ['actual_vacc_df', 'proj_vacc_df'] and val is not None:
                serial_dict[key] = self.serialize_vacc(val)
            elif key == 'actual_hosp' and val is not None:
                serial_dict[key] = self.serialize_hosp(val)
            else:
                serial_dict[key] = val
        return json.dumps(serial_dict)

    def from_json_string(self, s):
        logger.debug(f"{str(self.tags)} repopulating model from serialized json")
        # TODO: handle mobility, add in proj_mobility
        raw = json.loads(s)
        for key, val in raw.items():
            if val in ['_CovidModel__start_date', '_CovidModel__end_date']:
                self.__dict__[key] = dt.datetime.strptime(val, "%Y-%m-%d").date()
            elif key == "tc_cov":
                #TODO
                pass
            elif key in ['actual_vacc_df', 'proj_vacc_df'] and val is not None:
                self.__dict__[key] = CovidModel.unserialize_vacc(val)
            elif key == 'actual_hosp' and val is not None:
                self.__dict__[key] = CovidModel.unserialize_hosp(val)
            else:
                self.__dict__[key] = val

        # triggers updating of tmax, trange, etc.
        self.end_date = self.end_date

    def write_specs_to_db(self, engine):
        logger.debug(f"{str(self.tags)} writing specs to db")
        # returns all the data you would need to write to the database but doesn't actually write to the database

        with Session(engine) as session:
            # generate a spec_id so we can assign it to ourselves
            specs_table = get_sqa_table(engine, schema='covid_model', table='specifications')
            max_spec_id = session.query(func.max(specs_table.c.spec_id)).scalar()
            self.spec_id = max_spec_id + 1

            stmt = specs_table.insert().values(OrderedDict([
                ("base_spec_id", int(self.base_spec_id) if self.base_spec_id is not None else None),
                ("spec_id", self.spec_id),
                ("created_at", dt.datetime.now()),
                ("start_date", self.start_date),
                ("end_date", self.end_date),
                ("tags", json.dumps(self.tags)),
                ("regions", json.dumps(self.regions)),
                ("tslices", self.tc_tslices),
                ("tc", self.tc),
                ("serialized_model", self.to_json_string())
            ]))
            session.execute(stmt)
            session.commit()

    def read_from_base_spec_id(self, engine):
        df = pd.read_sql_query(f"select * from covid_model.specifications where spec_id = {self.base_spec_id}", con=engine, coerce_float=True)
        if len(df) == 0:
            ermsg = f'{self.base_spec_id} is not a valid spec ID.'
            logger.exception(f"{str(self.tags)}" + ermsg)
            raise ValueError(ermsg)
        row = df.iloc[0]
        self.from_json_string(row['serialized_model'])

    def _col_to_json(self, d):
        return json.dumps(d, ensure_ascii=False)

    def write_results_to_db(self, engine, new_spec=False, vals_json_attr='seir', cmpts_json_attrs=('region', 'age', 'vacc')):
        logger.debug(f"{str(self.tags)} writing results to db")
        table = 'results_v2'
        # if there's no existing spec_id assigned, write specs to db to get one
        if self.spec_id is None or new_spec:
            self.write_specs_to_db(engine)

        # build data frame with index of (t, region, age, vacc) and one column per seir cmpt
        solution_sum_df = self.solution_sum([vals_json_attr] + list(cmpts_json_attrs)).stack(cmpts_json_attrs)

        # build export dataframe
        df = pd.DataFrame(index=solution_sum_df.index)
        df['date'] = solution_sum_df.index.get_level_values('date')
        df['cmpt'] = solution_sum_df.index.droplevel('date').to_frame().to_dict(
            orient='records') if solution_sum_df.index.nlevels > 1 else None
        df['vals'] = solution_sum_df.to_dict(orient='records')
        for col in ['cmpt', 'vals']:
            df[col] = df[col].map(self._col_to_json)

        # if a sim_id is provided, insert it as a simulation result; some fields are different
        # build unique parameters dataframe
        df['created_at'] = dt.datetime.now()
        df['spec_id'] = self.spec_id
        df['result_id'] = pd.read_sql(f'select coalesce(max(result_id), 0) from covid_model.{table}', con=engine).values[0][0] + 1

        # write to database
        chunksize = int(np.floor(5000.0 / df.shape[1]))  # max parameters is 10,000. Assume 1 param per column and give some wiggle room because 10,000 doesn't always work

        results = df.to_sql(table, con=engine, schema='covid_model', index=False, if_exists='append', method='multi', chunksize=chunksize)

        self.result_id = df['result_id'][0]
        return df
