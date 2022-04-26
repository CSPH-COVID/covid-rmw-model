### Python Standard Library ###
from operator import attrgetter
import os
import json
from datetime import date
### Third Party Imports ###
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
### Local Imports ###
from covid_model import CovidModel, ModelSpecsArgumentParser, db_engine, CovidModelFit
from covid_model.run_scripts.run_solve_seir import run_solve_seir
from covid_model.run_scripts.run_fit import run_fit
from covid_model.utils import get_filepath_prefix

###################################################################################
# TODO: FIX THIS CODE; IT'S TOTALLY BROKEN
###################################################################################


def build_legacy_output_df(model: CovidModel):
    ydf = model.solution_sum(['seir', 'age']).stack(level='age')
    dfs_by_group = []
    for i, group in enumerate(model.attr['age']):
        dfs_by_group.append(ydf.xs(group, level='age').rename(columns={var: var + str(i+1) for var in model.attr['seir']}))
    df = pd.concat(dfs_by_group, axis=1)

    params_df = model.params_as_df
    combined = model.solution_ydf.stack(model.param_attr_names).join(params_df)

    totals = model.solution_sum('seir')
    totals_by_priorinf = model.solution_sum(['seir', 'priorinf'])
    df['Iht'] = totals['Ih']
    df['Dt'] = totals['D']
    df['Rt'] = totals_by_priorinf[('S', 'none')]
    df['Itotal'] = totals['I'] + totals['A']
    df['Etotal'] = totals['E']
    df['Einc'] = (combined['E'] / combined['alpha']).groupby('t').sum()
    # df['Einc'] = totals_by_variant * params_df / model.model_params['alpha']
    # for i, age in enumerate(model.attr['age']):
    #     df[f'Vt{i+1}'] = (model.solution_ydf[('S', age, 'vacc')] + model.solution_ydf[('R', age, 'vacc')]) * params_df.xs((age, 'vacc'), level=('age', 'vacc'))['vacc_eff']
    #     df[f'immune{i+1}'] = by_age[('R', age)] + by_age_by_vacc[('S', age, 'vacc')] * params_df.xs((age, 'vacc'), level=('age', 'vacc'))['vacc_eff']
    df['Vt'] = model.immunity(variant='omicron', vacc_only=True)
    df['immune'] = model.immunity(variant='omicron')
    df['date'] = model.daterange
    df['Ilag'] = totals['I'].shift(3)
    df['Re'] = model.re_estimates
    df['prev'] = 100000.0 * df['Itotal'] / model.model_params['total_pop']
    df['oneinX'] = model.model_params['total_pop'] / df['Itotal']
    df['Exposed'] = 100.0 * df['Einc'].cumsum()

    df.index.names = ['t']
    return df


def build_tc_df(model: CovidModel):
    return pd.DataFrame.from_dict({'time': model.tslices[:-1], 'tc_pb': model.efs, 'tc': model.obs_ef_by_slice})


def tags_to_scen_label(tags):
    if tags['run_type'] == 'Current':
        return 'Current Fit'
    elif tags['run_type'] == 'Prior':
        return 'Prior Fit'
    elif tags['run_type'] == 'Vaccination Scenario':
        return f'Vaccine Scenario: {tags["vacc_cap"]}'
    elif tags['run_type'] == 'TC Shift Projection':
        return f'TC Shift Scenario: {tags["tc_shift"]} on {tags["tc_shift_date"]} ({tags["vacc_cap"]})'


def run_model(model, engine, legacy_output_dict=None):
    print('Scenario tags: ', model.tags)
    model.solve_seir()
    model.write_results_to_db(engine, new_spec=True)
    if legacy_output_dict is not None:
        legacy_output_dict[tags_to_scen_label(model.tags)] = build_legacy_output_df(model)



def run_model_scenarios(params_scens, vacc_proj_params_scens, mobility_proj_params_scens,
                        attribute_multipliers_scens, outdir, fname_extra="", refit_from_date=None, fit_args=None, **specs_args):
    if (outdir):
        os.makedirs(outdir, exist_ok=True)
    engine = db_engine()

    # compile scenarios:
    scens_files = [json.load(open(sf, 'r')) if sf is not None else None for sf in [params_scens, vacc_proj_params_scens, mobility_proj_params_scens, attribute_multipliers_scens]]
    scens = [key for sf in scens_files if sf is not None for key in sf.keys()]

    # initialize Base model:
    base_model = CovidModel(engine=engine, **specs_args)

    ms = {}
    dfs = []
    dfhs = []
    for scen in scens:
        print(f"Scenario: {scen}: Copying / Modifying Model")
        scen_base_model = CovidModel(engine=engine, base_model=base_model)
        # Update params based on scenario
        if scens_files[0] and scen in scens_files[0]:
            scen_base_model.params.update(scens_files[0][scen])
        if scens_files[1] and scen in scens_files[1]:
            scen_base_model.vacc_proj_params.update(scens_files[1][scen])
        if scens_files[2] and scen in scens_files[2]:
            scen_base_model.mobility_proj_params.update(scens_files[2][scen])
        # Note: attribute multipliers is a list, so our easiest option is to append. So you probably want to remove
        # something from the base attribute multipliers file if the scenarios are exploring different settings for those.
        if scens_files[3] and scen in scens_files[3]:
            scen_base_model.attribute_multipliers.extend(scens_files[3][scen])
        scen_model = CovidModel(base_model=scen_base_model)
        if refit_from_date is not None:
            fit_args['look_back'] = len([t for t in scen_model.tslices if t >= (refit_from_date - scen_model.start_date).days])
            fit = CovidModelFit(engine=engine, tc_min=fit_args['tc_min'], tc_max=fit_args['tc_max'], from_specs=scen_model)
            fit.set_actual_hosp(engine=engine, county_ids=scen_model.get_all_county_fips())
            fit.run(engine, **fit_args, print_prefix=f'{scen}', outdir=outdir)
            scen_model.apply_tc(fit.fitted_model.tc, fit.fitted_model.tslices)
        print(f"Scenario: {scen}: Prepping and Solving SEIR")
        scen_model, df, dfh = run_solve_seir(outdir=outdir, model=scen_model, prep_model=True, tags={'scenario': scen})
        ms[scen] = scen_model
        dfs.append(df.assign(scen=scen))
        dfhs.append(dfh.assign(scen=scen))

    df = pd.concat(dfs, axis=0).set_index(['scen', 'date', 'region', 'seir'])
    dfh = pd.concat(dfhs, axis=0)
    dfh_measured = dfh[['currently_hospitalized']].rename(columns={'currently_hospitalized': 'hospitalized'}).loc[dfh['scen'] == scens[0]].assign(series='observed')
    dfh_modeled = dfh[['modeled_hospitalized', 'scen']].rename(columns={'modeled_hospitalized': 'hospitalized', 'scen': 'series'})
    dfh2 = pd.concat([dfh_measured, dfh_modeled], axis=0).set_index('series', append=True)

    print("saving results")
    df.to_csv(get_filepath_prefix(outdir) + f"run_model_scenarios_compartments_{fname_extra}.csv")
    dfh.to_csv(get_filepath_prefix(outdir) + f"run_model_scenarios_hospitalized_{fname_extra}.csv")
    dfh2.to_csv(get_filepath_prefix(outdir) + f"run_model_scenarios_hospitalized2_{fname_extra}.csv")

    print("plotting results")
    p = sns.relplot(data=df, x='date', y='y', hue='scen', col='region', row='seir', kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + f"run_model_scenarios_compartments_{fname_extra}.png", dpi=300)

    p = sns.relplot(data=dfh2, x='date', y='hospitalized', hue='series', col='region', col_wrap=min(3, len(specs_args['regions'])), kind='line', facet_kws={'sharex': False, 'sharey': False}, height=2, aspect=4)
    _ = [ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())) for ax in p.axes.flat]
    plt.savefig(get_filepath_prefix(outdir) + f"run_model_scenarios_hospitalized_{fname_extra}.png", dpi=300)

    print("done")

    return df, dfh, dfh2, ms


if __name__ == '__main__':
    outdir = os.path.join("covid_model", "output", os.path.basename(__file__))

    parser = ModelSpecsArgumentParser()
    parser.add_argument("-psc", '--params_scens', type=str, help="path to parameters scenario file to use (updates base model parameters)")
    parser.add_argument("-vppsc", '--vacc_proj_params_scens', type=str, help="path to vaccine projection parameters scenario file (updates base vpp)")
    parser.add_argument("-mppsc", '--mobility_proj_params_scens', type=str, help="path to mobility projection parameters scenario file (updates base mpp)")
    parser.add_argument('-amsc', '--attribute_multipliers_scens', type=str, help="path to attribute multipliers scenario file (updates base mprev)")
    parser.add_argument('-rfd', '--refit_from_date', type=date.fromisoformat, help="refit from this date forward for each scenario, or don't refit if None (format: YYYY-MM-DD)")
    parser.add_argument("-fne", '--fname_extra', default="", help="extra info to add to all files saved to disk")

    specs_args = parser.specs_args_as_dict()
    non_specs_args = parser.non_specs_args_as_dict()

    # note refitting doesn't work from CLI because we aren't collecting fit specs here. Better way to do this?

    run_model_scenarios(**non_specs_args, outdir=outdir, **specs_args)
