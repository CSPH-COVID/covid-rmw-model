""" Python Standard Library """
import copy
import itertools
import os
from multiprocessing import Pool
import logging
import json
from time import perf_counter
from itertools import pairwise
import datetime as dt
from scipy.stats import norm as sp_norm

""" Third Party Imports """
from multiprocessing_logging import install_mp_handler
import pandas as pd
import numpy as np
from scipy import optimize as spo
from matplotlib import pyplot as plt
from matplotlib.cm import tab10, tab20b, tab20c, tab20
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
import seaborn as sns
""" Local Imports """
from covid_model import RMWCovidModel
from covid_model.analysis.charts import plot_transmission_control
from covid_model.utils import IndentLogger, setup, get_filepath_prefix, db_engine
from covid_model.analysis.charts import plot_modeled, plot_observed_hosps, plot_variant_proportions, format_date_axis
logger = IndentLogger(logging.getLogger(''), {})


def __single_batch_fit(model: RMWCovidModel, tc_min, tc_max, yd_start=None, tstart=None, tend=None, regions=None):
    """function to fit TC for a single batch of time for a model
    Only TC values which lie in the specified regions between tstart and tend will be fit.
    Args:
        model: model to fit
        tc_min: minimum allowable TC
        tc_max: maximum allowable TC
        yd_start: initial conditions for the model at tstart. If None, then model's y0_dict is used.
        tstart: start time for this batch
        tend: end time for this batch
        regions: regions which should be fit. If None, all regions will be fit
    Returns: Fitted TC values and the estimated covariance matrix between the different TC values.
    """
    # define initial states
    regions = model.regions if regions is None else regions
    tc = {t: model.tc[t] for t in model.tc.keys() if tstart <= t <= tend}
    tc_ts  = list(tc.keys())
    yd_start = model.y0_dict if yd_start is None else yd_start
    y0 = model.y0_from_dict(yd_start)
    trange = range(tstart, tend+1)
    # hrf_finder
    # To take out hrf: change 'estimated_actual' to 'observed':
    ydata = model.hosps.loc[pd.MultiIndex.from_product([regions, [model.t_to_date(t) for t in trange]])]['observed'].to_numpy().flatten('F')
    # Min-max scaling (scales to [0,1])
    max_scale = ydata.max()
    ydata = ydata / max_scale
    def tc_list_to_dict(tc_list):
        """convert tc output of curve_fit to a dict like in our model.
        curve_fit assumes you have a function which accepts a vector of inputs. So it will provide guesses for TC as a
        vector. We need to convert that vector to a dictionary in order to update the model.
        Args:
            tc_list: the list of tc values to update.
        Returns: dictionary of TC suitable to pass to the model.
        """
        i = 0
        tc_dict = {t: {} for t in tc_ts}
        for tc_t in tc.keys():
            for region in regions:
                tc_dict[tc_t][region] = tc_list[i]
                i += 1
        return tc_dict

    def tc_cov_mat_to_dict(tc_cov):
        """Converts fitted TC covariance matrix into a dict structure"""
        i = 0
        tc_cov_dict = {t: {r: {} for r in regions} for t in tc.keys()}
        for tc_i in tc.keys():
            for r_i in regions:
                j = 0
                for tc_j in tc.keys():
                    for r_j in regions:
                        tc_cov_dict[tc_i][r_i].update({tc_j: {r_j: tc_cov[i, j]}})
                        j += 1
                i += 1
        return tc_cov_dict

    def func(trange, *test_tc):
        """A simple wrapper for the model's solve_seir method so that it can be optimzed by curve_fit
        Args:
            trange: the x values of the curve to be fit. necessary to match signature required by curve_fit, but not used b/c we already know the trange for this batch
            *test_tc: list of TC values to try for the model
        Returns: hospitalizations for the regions of interest for the time periods of interest.
        """
        model.update_tc(tc_list_to_dict(test_tc), replace=False, update_lookup=False)
        model.solve_seir(y0=y0, tstart=tstart, tend=tend)
        return model.solution_sum_Ih(tstart, tend, regions=regions)/max_scale
    fitted_tc, fitted_tc_cov, info_dict, *other = spo.curve_fit(
        f=func,
        xdata=trange,
        ydata=ydata,
        p0=[tc[t][region] for t in tc_ts for region in model.regions],
        bounds=([tc_min] * len(tc_ts) * len(regions), [tc_max] * len(tc_ts) * len(regions)),
        verbose=2,
        full_output=True)
    fitted_tc = tc_list_to_dict(fitted_tc)
    # Standard deviation of the estimates for each TC is the square root of the diagonal elements (variances) of each
    # estimated TC parameter.
    fitted_tc_cov = tc_cov_mat_to_dict(fitted_tc_cov)
    return fitted_tc, fitted_tc_cov


def __optimize_variants(model: RMWCovidModel, variants:list, tstart:int, tend:int, regions=None, yd_start=None):
    # define initial states
    regions = model.regions if regions is None else regions
    yd_start = model.y0_dict if yd_start is None else yd_start
    y0 = model.y0_from_dict(yd_start)
    trange = range(tstart, tend+1)
    # hrf_finder
    # To take out hrf: change 'estimated_actual' to 'observed':
    n_variants = len(variants)
    ydata = model.variant_props.loc[trange,variants].to_numpy().flatten("F")

    def variant_func(params,*args):
        """A simple wrapper for the model's solve_seir method so that it can be optimzed by curve_fit

        Args:
            trange: the x values of the curve to be fit. necessary to match signature required by curve_fit, but not used b/c we already know the trange for this batch
            *test_tc: list of TC values to try for the model

        Returns: hospitalizations for the regions of interest for the time periods of interest.

        """
        offsets = params[:n_variants]
        scales = params[n_variants:]
        model.update_seed_offsets({f"{variant}_seed":offset for variant, offset in zip(variants,offsets)})
        model.update_seed_scalers({f"{variant}_seed":scale for variant, scale in zip(variants,scales)})
        model.solve_seir(y0=y0, tstart=tstart, tend=tend)
        return np.sum(np.square(model.solution_var_props(tstart,tend,variants=variants) - ydata))

    def progress_log(xk, convergence):
        logger.info(f"{str(model.tags)}: Diff Ev: xk={xk}")

    opt_result = spo.differential_evolution(func=variant_func,
                                            bounds=[(-model.voffset_max,model.voffset_max) for _ in range(len(variants))] +
                                                   [(0,model.soffset_max) for _ in range(len(variants))],
                                            integrality=[True]*len(variants) + [False]*len(variants),
                                            x0=[model.seed_offsets[f"{variant}_seed"] for variant in variants] +
                                               [model.seed_scalers[f"{variant}_seed"] for variant in variants],
                                            disp=True,
                                            polish=False,
                                            callback=progress_log)

    return opt_result


def __single_batch_fit_variant_opt(model: RMWCovidModel, tc_min, tc_max, relevant_variants, yd_start=None, model_start_t = None,  tstart=None, tend=None, regions=None):
    """function to fit TC for a single batch of time for a model

    Only TC values which lie in the specified regions between tstart and tend will be fit.

    Args:
        model: model to fit
        tc_min: minimum allowable TC
        tc_max: maximum allowable TC
        yd_start: initial conditions for the model at tstart. If None, then model's y0_dict is used.
        tstart: start time for this batch
        tend: end time for this batch
        regions: regions which should be fit. If None, all regions will be fit

    Returns: Fitted TC values and the estimated covariance matrix between the different TC values.

    """
    # define initial states
    regions = model.regions if regions is None else regions
    tc = {t: model.tc[t] for t in model.tc.keys() if tstart <= t <= tend}
    tc_ts = list(tc.keys())
    yd_start = model.y0_dict if yd_start is None else yd_start
    y0 = model.y0_from_dict(yd_start)
    #y0 = model.y0_from_dict(model.y0_dict)
    trange = range(tstart, tend+1)
    # hrf_finder
    # To take out hrf: change 'estimated_actual' to 'observed':
    #ydata = np.log(1+model.hosps.loc[pd.MultiIndex.from_product([regions, [model.t_to_date(t) for t in trange]])]['observed'].to_numpy().flatten('F'))

    ydata = model.hosps.loc[pd.MultiIndex.from_product([regions, [model.t_to_date(t) for t in trange]])]['observed'].to_numpy().flatten('F')
    max_scale = ydata.max()
    ydata = ydata / max_scale
    ydata_variants = model.variant_props.loc[trange]
    #relevant_variants = [c for c in ydata_variants if (ydata_variants[c]!=0).any()]
    n_relevant_variants = len(relevant_variants)
    ydata_variants_reduced = ydata_variants.loc[:,relevant_variants].to_numpy().flatten("F")
    ydata = np.concatenate([ydata,ydata_variants_reduced])
    def tc_list_to_dict(tc_list):
        """convert tc output of curve_fit to a dict like in our model.

        curve_fit assumes you have a function which accepts a vector of inputs. So it will provide guesses for TC as a
        vector. We need to convert that vector to a dictionary in order to update the model.

        Args:
            tc_list: the list of tc values to update.

        Returns: dictionary of TC suitable to pass to the model.

        """
        i = 0
        tc_dict = {t: {} for t in tc_ts}
        for tc_t in tc.keys():
            for region in regions:
                tc_dict[tc_t][region] = tc_list[i]
                i += 1
        return tc_dict

    def tc_func(trange, *params):
        """A simple wrapper for the model's solve_seir method so that it can be optimzed by curve_fit

        Args:
            trange: the x values of the curve to be fit. necessary to match signature required by curve_fit, but not used b/c we already know the trange for this batch
            *test_tc: list of TC values to try for the model

        Returns: hospitalizations for the regions of interest for the time periods of interest.

        """
        # Extract parameters
        test_tc = params[:-2*n_relevant_variants]
        var_seed_offsets = params[len(test_tc):-n_relevant_variants]
        var_seed_scalers = params[len(test_tc)+n_relevant_variants:]
        # Convert offsets to integers
        #var_offsets_int = [int(np.round(v * voffset_max)) for v in var_seed_offsets]
        # Update offsets
        model.update_seed_offsets({f"{variant}_seed": offset for variant, offset in zip(relevant_variants, var_seed_offsets)})
        # Update scaling values
        model.update_seed_scalers({f"{variant}_seed": scaler for variant, scaler in zip(relevant_variants, var_seed_scalers)})
        # Update TC
        model.update_tc(tc_list_to_dict(test_tc), replace=False, update_lookup=False)
        # Solve the model
        model.solve_seir(y0=y0, tstart=tstart, tend=tend)

        ypred = np.concatenate([
            #np.log(1+model.solution_sum_Ih(tstart, tend, regions=regions)),
            model.solution_sum_Ih(tstart,tend,regions=regions)/max_scale,
            model.solution_var_props(tstart,tend,variants=relevant_variants)
        ])

        return ypred


    fitted_p, fitted_p_cov = spo.curve_fit(
        f=tc_func,
        xdata=trange,
        ydata=ydata,
        p0=[tc[t][region] for t in tc_ts for region in model.regions] +  # TC guesses
           ([model.seed_offsets[f"{variant}_seed"] for variant in relevant_variants]) +  # Offset guesses
           ([model.seed_scalers[f"{variant}_seed"] for variant in relevant_variants]), # Scaler guesses
        bounds=(
            ([tc_min] * len(tc_ts) * len(regions)) +  # TC min bound
            ([-model.voffset_max] * n_relevant_variants) +  # Offset min bound
            ([0.0] * n_relevant_variants), # Scaler min bound
            ([tc_max] * len(tc_ts) * len(regions)) +
            ([model.voffset_max] * n_relevant_variants) +
            ([model.soffset_max] * n_relevant_variants)
        ),
        verbose=2
    )

    logger.info(f"{str(model.tags)}: Optimization finished.")

    # res = spo.direct(
    #     func=loss,
    #     # x0=[tc[t][region] for t in tc_ts for region in model.regions] +  # TC guesses
    #     #    [model.seed_offsets[f"{variant}_seed"]/model.voffset_max for variant in relevant_variants] +  # Offset guesses
    #     #    [model.seed_scalers[f"{variant}_seed"]/model.soffset_max for variant in relevant_variants], # Scaler guesses
    #     bounds=([(tc_min,tc_max)] * len(tc_ts) * len(regions)) +  # TC min bound
    #            ([(-1.0,1.0)] * n_relevant_variants) +  # Offset min bound
    #            ([(0.0,1.0)] * n_relevant_variants),# Scaler min bound
    # )
    # if not res.success:
    #     raise RuntimeError("Optimization failed.")

    return fitted_p, fitted_p_cov


def set_tc_for_projection(model: RMWCovidModel, last_n_tc: int = 4):
    """ Sets the last TC value to the average of last_n_tc values (used for projection).
    :param model: The model to operate on.
    :param last_n_tc:
    :return: Dictionary of original last TC values for each region, to restore later if more fitting will occur.
    """
    keys_sorted = sorted(list(model.tc.keys()))
    last_n_keys = keys_sorted[-last_n_tc:]

    last_tc_dict = {}
    for region in model.attrs["region"]:
        avg_tc = np.mean([model.tc[k][region] for k in last_n_keys])
        last_tc_dict[region] = model.tc[last_n_keys[-1]][region]
        model.tc[last_n_keys[-1]][region] = avg_tc

    return last_tc_dict


def restore_tc(model: RMWCovidModel, last_tc_dict):
    """ Restores TC values from last_tc_dict, if the last TC value was set for projection.
    :param model: The model to operate on.
    :param last_tc_dict: Dictionary keyed by region, containing tc values for the last T key.
    :return: None
    """
    last_key = max(model.tc.keys())

    model.tc[last_key].update(last_tc_dict)


def forward_sim_plot(model, outdir, highlight_range=None, n_sims=None, last_n=4):
    """Solve the model's ODEs and plot transmission control, and save hosp & TC data to disk

    Args:
        model: the model to solve and plot.
    """
    # TODO: refactor into charts.py?
    logger.info(f'{str(model.tags)}: Running forward sim')
    fig, axs = plt.subplots(nrows=3, ncols=len(model.regions), figsize=(10*len(model.regions), 18), dpi=300, sharex=True, sharey=False, squeeze=False)
    # If we are running TC simulations
    sim_tc_dict = None
    if n_sims is not None:
        rng = np.random.default_rng(42)
        last_n_keys = sorted(model.tc.keys())[-last_n:]
        n_keys = len(last_n_keys)
        avg_tcs = []
        avg_tcs_var = []
        for region in model.regions:
            avg_tc = np.mean([model.tc[k][region] for k in last_n_keys])
            avg_tcs.append(avg_tc)
            # Covariance of a mean of random variables:
            # https://stats.stackexchange.com/questions/168971/variance-of-an-average-of-random-variables
            var_avg_tc = (np.sum([model.tc_cov[k][region][k][region] for k in last_n_keys]) +
                          2 * np.sum([model.tc_cov[i][region][j][region] for i in last_n_keys for j in last_n_keys if j < i]))
            var_avg_tc = var_avg_tc/(n_keys**2)
            avg_tcs_var.append(var_avg_tc)
        sim_tc = rng.normal(loc=avg_tcs, scale=np.sqrt(avg_tcs_var), size=(n_sims, len(model.regions))).transpose()
        sim_tc_dict = {r: s for r, s in zip(model.regions, sim_tc)}
    hosps_df = model.modeled_vs_observed_hosps()
    for i, region in enumerate(model.regions):
        # Observed hospitalizations are plotted in their entirety.
        axs[0, i].plot(hosps_df.loc[region,"observed"], label="Observed Hosp.", color=tab10(0))
        # Split modeled hospitalizations so that we can delineate when fitting ends and projection starts
        region_modeled_hosps = hosps_df.loc[region, "modeled_observed"]
        fit_end = hosps_df.loc[region, "observed"].isna().idxmax()
        fitted_hosps = region_modeled_hosps.loc[:fit_end]
        axs[0, i].plot(fitted_hosps, label="Modeled Hosp. (Fitted)", color=tab10(1))
        if n_sims is not None:
            last_key = last_n_keys[-1]
            #orig_tc_dict = copy.deepcopy(model.tc)
            sim_df = {}
            for s_i, sim_tc in enumerate(sim_tc_dict[region]):
                model.update_tc({last_key+1: {region: sim_tc}}, replace=False)
                model.solve_seir(tstart=last_key, y0=model.solution_y[last_key-1])
                sim_df[s_i] = model.solution_sum_Ih()[last_key:]
                logger.info(f"{str(model.tags)}: Finished {s_i+1} of {n_sims} simulations.")
            model.update_tc({last_key+1: {region: avg_tcs[i]}}, replace=False)
            sim_df = pd.DataFrame(sim_df)

            xmin = min(sim_tc_dict[region]) - 0.02
            xmax = max(sim_tc_dict[region]) + 0.02

            norm_pdf_x = np.linspace(xmin, xmax, 1000)
            norm_pdf = sp_norm(loc=avg_tcs[i], scale=np.sqrt(avg_tcs_var[i])).pdf

            ds_fig, ds_ax = plt.subplots(figsize=(10,10))
            ds_ax.hist(sim_tc_dict[region], density=True, bins="auto", color=tab20(1), label="Sampled TC Histogram")
            ds_ax.plot(norm_pdf_x, norm_pdf(norm_pdf_x), linewidth=3, color=tab20(0), label=f"PDF of N({avg_tcs[i]},{avg_tcs_var[i]})")
            ds_ax.set_xlim(xmin,xmax)
            ds_ax.set_ylabel("Relative Likelihood")
            ds_ax.legend(fancybox=False, edgecolor="black")
            ds_ax.set_title("Distribution of Sampled TC")
            ds_fig.tight_layout()
            ds_fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + f"_{region}_tc_dist.png")
            plt.close(ds_fig)
            sim_df_tmp = sim_df.set_index(pd.date_range(model.t_to_date(last_key), periods=len(sim_df)))
            sim_df_tmp.to_csv(get_filepath_prefix(outdir, tags=model.tags) + f"_{region}_sim_df.csv")
            # first_sim = True
            # for c in sim_df_tmp:
            #     axs[0,i].plot(sim_df_tmp[c], label="Modeled Hosp. (Simulated)" if first_sim else None, alpha=0.05, color=tab10(1), linewidth=0.5)
            #     first_sim = False
            #
            # axs[0, i].plot(sim_df_tmp.mean(axis=1), label="Modeled Hosp. (Mean Simulated)", color=tab10(1),
            #                linestyle="--")

            sns_df = sim_df_tmp.melt(ignore_index=False, value_name="sim_hosps",
                                     var_name="sim_num").reset_index().rename(columns={"index": "date"})

            sns.lineplot(data=sns_df, x="date", y="sim_hosps", errorbar="pi", ax=axs[0, i], color=tab10(1),
                         linestyle="--", label="Modeled Hosp (Simulated)")

        else:
            proj_hosps = region_modeled_hosps.loc[fit_end:]
            if len(proj_hosps) > 0:
                axs[0, i].plot(proj_hosps, label="Modeled Hosp. (Projection)", color=tab10(1), linestyle="--")
        #hosps_df.loc[region].plot(ax=axs[0, i])
        axs[0,i].title.set_text(f'Hospitalizations: {region}')
        axs[0,i].legend(fancybox=False, edgecolor="black")
        plot_transmission_control(model, [region], ax=axs[1,i])
        if highlight_range is not None:
            lb,ub = highlight_range
            axs[0,i].axvspan(xmin=lb,xmax=ub,color="gray",alpha=0.5)
        axs[1, i].title.set_text(f'TC: {region}')
        plot_variant_proportions(model,ax=axs[2,i])
        #axs[2,i].legend()
    fig.tight_layout()
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + '_model_fit.png')
    if n_sims is not None:
        for ax_i in range(len(model.regions)):
            axs[0,ax_i].set_xlim(pd.to_datetime("2023-01-01"), max(sim_df_tmp.index))
        fig.tight_layout()
        fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + "_model_fit_2023.png")
    plt.close(fig)
    hosps_df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + '_model_fit.csv')
    json.dump(dict(model.tc), open(get_filepath_prefix(outdir, tags=model.tags) + '_model_tc.json', 'w'))

def tc_simulation(model: RMWCovidModel, n_sims: int = 100):
    """ Sample from the distribution of the last fitted value of TC and simulate possible outcomes.
    :param model: Model to simulate
    :return: None
    """
    rng = np.random.default_rng(1)

    tc_sd = model.tc_cov

    last_tc_key = max(model.tc.keys())

    for region in model.attrs["region"]:
        tc_mean = model.tc[last_tc_key][region]
        tc_std = model.tc_cov[last_tc_key][region]

        # Sample
        samples = rng.normal(loc=tc_mean, scale=tc_std)


def do_single_fit(tc_0=0.75,
                  tc_min=0,
                  tc_max=0.99,
                  tc_window_size=14,
                  tc_window_batch_size=6,
                  tc_batch_increment=2,
                  last_tc_window_min_size=21,
                  fit_start_date=None,
                  fit_end_date=None,
                  prep_model=True,
                  pickle_matrices=True,
                  pre_solve_model=False,
                  outdir=None,
                  write_results=True,
                  write_batch_results=False,
                  model_class=RMWCovidModel,
                  **model_args):
    """ Fits TC for the model between two dates, and does the fit in batches to make the solving easier

    Args:
        tc_0: default value for TC
        tc_min: minimum allowable TC
        tc_max: maximum allowable TC
        tc_window_size: How often to update TC (days)
        tc_window_batch_size: How many windows to fit at once
        tc_batch_increment: How many TC windows to shift over for each batch fit
        last_tc_window_min_size: smallest size of the last TC window
        fit_start_date: refit all tc's on or after this date (if None, use model start date)
        fit_end_date: refit all tc's up to this date (if None, uses either model end date or last date with hospitalization data, whichever is earlier)
        prep_model: Should we run model.prep before fitting (useful if the model_args specify a base_model which has already been prepped)
        pickle_matrices: Should we pickle the ODE matrices and write to a file. The default is True, but setting to False is useful if we are running on Google Cloud.
        pre_solve_model: Should we run model.solve_seir() before fitting (useful if the fit_start_date is after model.start_date and so we need an initial solution to get the initial conditions
        outdir: the output directory for saving results
        write_results: should final results be written to the database
        write_batch_results: Should we write output to the database after each fit
        model_class: What class to use for the CovidModel. Useful if using different versions of the model (with more or fewer compartments, say)
        **model_args: Arguments to be used in creating the model to be fit.

    Returns: Fitted model

    """


    logging.debug(str({"model_build_args": model_args}))

    # get a db connection if we're going to be writing results
    if write_batch_results or write_results:
        engine = db_engine()

    model = model_class(**model_args)

    # adjust fit start and end, and check for consistency with model and hosp dates
    fit_start_date = dt.datetime.strptime(fit_start_date, '%Y-%m-%d').date() if fit_start_date is not None and isinstance(fit_start_date, str) else fit_start_date
    fit_end_date = dt.datetime.strptime(fit_end_date, '%Y-%m-%d').date() if fit_end_date is not None and isinstance(fit_end_date, str) else fit_end_date
    fit_start_date = model.start_date if fit_start_date is None else fit_start_date
    fit_end_date = min(model.end_date, model.hosps.index.get_level_values(1).max()) if fit_end_date is None else fit_end_date
    ermsg = None
    if fit_start_date < model.start_date:
        ermsg = f'Fit needs to start on or after model start date. Opted to start fitting at {fit_start_date} but model start date is {model.start_date}'
    elif fit_end_date > model.end_date:
        ermsg = f'Fit needs to end on or before model end date. Opted to stop fitting at {fit_end_date} but model end date is {model.end_date}'
    elif fit_end_date > model.hosps.index.get_level_values(1).max():
        ermsg = f'Fit needs to end on or before last date with hospitalization data. Opted to stop fitting at {fit_end_date} but last date with hospitalization data is {model.hosps.index.get_level_values(1).max()}'
    if ermsg is not None:
        logger.exception(f"{str(model.tags)}" + ermsg)
        raise ValueError(ermsg)

    # prep model (we only do this once to save time)
    if prep_model:
        logger.info(f'{str(model.tags)} Prepping Model')
        t0 = perf_counter()
        model.prep(outdir=outdir,pickle_matrices=pickle_matrices)
        logger.debug(f'{str(model.tags)} Model flows {model.flows_string}')
        logger.info(f'{str(model.tags)} Model prepped for fitting in {perf_counter() - t0} seconds.')

    # prep model (we only do this once to save time)
    if pre_solve_model:
        logger.info(f'{str(model.tags)} Solving Model ODE')
        t0 = perf_counter()
        model.seir_graph_trace()
        logger.info(f'{str(model.tags)} Model solved in {perf_counter() - t0} seconds.')

    # replace the TC and tslices within the fit window
    fit_tstart = model.date_to_t(fit_start_date)
    fit_tend = model.date_to_t(fit_end_date)
    if tc_0 is not None:
        tc = {t: tc for t, tc in model.tc.items() if t < fit_tstart or t > fit_tend}
        tc.update({t: {region: tc_0 for region in model.regions} for t in range(fit_tstart, fit_tend - last_tc_window_min_size, tc_window_size)})
        model.update_tc(tc)

    # Get start/end for each batch
    relevant_tc_ts = [t for t in model.tc.keys() if fit_tstart <= t <= fit_tend]
    last_batch_start_index = -min(tc_window_batch_size, len(relevant_tc_ts))
    batch_tstarts = relevant_tc_ts[:last_batch_start_index:tc_batch_increment] + [relevant_tc_ts[last_batch_start_index]]
    batch_tends = [t - 1 for t in relevant_tc_ts[tc_window_batch_size::tc_batch_increment]] + [fit_tend]

    logger.info(f'{str(model.tags)} Will fit {len(batch_tstarts)} times')
    for i, (tstart, tend) in enumerate(zip(batch_tstarts, batch_tends)):
        t0 = perf_counter()
        yd_start = model.y_dict(tstart) if tstart != 0 else model.y0_dict
        # batch_relevant_variants = list(model.variant_props.columns[np.any(~np.isclose(model.variant_props.iloc[tstart:tend],0.0),axis=0)])
        # fitted_tc, fitted_tc_cov = __single_batch_fit_variant_opt(model,
        #                                                           tc_min=tc_min,
        #                                                           tc_max=tc_max,
        #                                                           yd_start=yd_start,
        #                                                           relevant_variants=batch_relevant_variants,
        #                                                           tstart=0,
        #                                                           tend=tend)

        fitted_tc, fitted_tc_cov = __single_batch_fit(model,
                                                     tc_min=tc_min,
                                                     tc_max=tc_max,
                                                     yd_start=yd_start,
                                                     tstart=tstart,
                                                     tend=tend)
        # Update TC standard deviation estimates.
        model.tc_cov.update(fitted_tc_cov)
        model.tags['fit_batch'] = str(i)

        logger.info(f'{str(model.tags)}: Transmission control fit {i + 1}/{len(batch_tstarts)} completed in {perf_counter() - t0} seconds: {fitted_tc}')

        if write_batch_results:
            logger.info(f'{str(model.tags)}: Uploading batch results')
            model.write_specs_to_db(engine)
            logger.info(f'{str(model.tags)}: spec_id: {model.spec_id}')

        # Solve the model from the start until present to fix issue with TC window overlap.
        model.solve_seir(tstart=model.tstart, tend=tend)

        # simulate the model and save a picture of the output
        forward_sim_plot(model, outdir)

    model.tags['run_type'] = 'fit'
    logger.info(f'{str(model.tags)}: fitted TC: {model.tc}')
    logger.info(f'{str(model.tags)}: fitted Offsets: {model.seed_offsets}')
    logger.info(f'{str(model.tags)}: fitted Scalers: {model.seed_scalers}')

    # if outdir is not None:
    #     forward_sim_plot(model, outdir)

    if write_results:
        logger.info(f'{str(model.tags)}: Uploading final results')
        model.write_specs_to_db(engine)
        #model.write_results_to_db(engine)
        logger.info(f'{str(model.tags)}: spec_id: {model.spec_id}')
    # TODO: Change this to a function argument.
    # if True:
    #     logger.info(f'{str(model.tags)}: Optimizing variant seeds')
    #     relevant_variants = model.attrs["variant"][1:]
    #     for variant in relevant_variants:
    #         print(f"BF Optimizing Variant {variant}")
    #         os.makedirs(f"variant_test/{variant}/", exist_ok=True)
    #         for offset in range(-5, 5):
    #             fig, ax = plt.subplots(figsize=(15, 18), nrows=2)
    #             model.update_seed_offsets({f"{variant}_seed": offset})
    #             model.solve_seir()
    #             plot_variant_proportions(model, ax=ax[0], show_seeds=[variant])
    #             hosps_df = model.modeled_vs_observed_hosps()
    #             hosps_df.loc[model.regions[0]].plot(ax=ax[1])
    #             plot_observed_hosps(model, ax=ax[1])
    #             plt.savefig(f"variant_test/{variant}/{variant}_{offset:03d}")
    #             plt.close(fig)
    #         model.update_seed_offsets({f"{variant}_seed": 0})
    return model


def do_variant_optimization_ie(model: RMWCovidModel, outdir:str,  tc_min:float=0.0, tc_max:float=0.99, write_specs=True, **kwargs):
    model.solve_seir()
    model.tags["vopt"] = "pre_opt"
    logger.info(f"{str(model.tags)} Generating pre-optimization plot for comparison...")
    forward_sim_plot(model, outdir)
    for v1,v2 in pairwise(model.attrs["variant"]):
        if v1 == "none":
            continue
        logger.info(f"{str(model.tags)} Starting optimization process...")
        model.tags["vopt"] = f"{v1}_{v2}"
        start_t = perf_counter()
        # A variant's optimization region start when seeding begins and ends when the variant's proportion of infections
        # is close to 0

        vwindow_start = int(np.clip(
            model.seeds[model.regions[0]][f"{v1}_seed"].argmax() - model.voffset_max - 1, # Give us enough space so that we can optimize the seed backwards up to the limit.
            model.tstart,
            model.tend
        ))
        # The window ends when the variant proportion falls to 0.
        vwindow_end = int(len(model.variant_props) - np.argmax(~np.isclose(model.variant_props[v2].values[::-1],0.0)))
        # Clip the window end so that we do not surpass the end of hospitalizations
        hosps_end_t = model.date_to_t(model.hosps.index.get_level_values("date").max())
        vwindow_end = min(vwindow_end,model.tend,hosps_end_t)
        vwindow_ydict = model.y_dict(vwindow_start) if vwindow_start != 0 else model.y0_dict

        logger.info(f"{str(model.tags)}: Variant pair ({v1}, {v2}) appears between "
                    f"{model.t_to_date(vwindow_start)} and {model.t_to_date(vwindow_end)}")

        # Iterate fitting the variant seeds and fitting TC.
        # opt_result = __optimize_variants(model=model,
        #                                  variants=[v1,v2],
        #                                  tstart=vwindow_start,
        #                                  tend=vwindow_end,
        #                                  yd_start=vwindow_ydict)
        # model.tags["iter"] = f"{i+1}of{n_iters}"
        # model.tags["opt"] = "post_var"
        # forward_sim_plot(model,outdir=outdir)
        # fitted_tc_, fitted_tc_cov = __single_batch_fit(model=model,
        #                                               tc_min=tc_min,
        #                                               tc_max=tc_max,
        #                                               yd_start=vwindow_ydict,
        #                                               tstart=vwindow_start,
        #                                               tend=vwindow_end)
        # model.tags["opt"] = "post_tc"
        # forward_sim_plot(model,outdir=outdir)
        fitted_p, fitted_p_cov = __single_batch_fit_variant_opt(model,
                                                                tc_min=tc_min,
                                                                tc_max=tc_max,
                                                                relevant_variants=[v1, v2],
                                                                yd_start=vwindow_ydict,
                                                                tstart=vwindow_start,
                                                                tend=vwindow_end)
        fitted_tc = fitted_p[:-4]
        #fitted_offset = fitted_p[-4:-2]
        #current_offset = model.seed_offsets[f"{v1,v2}_seed"]
        #fitted_scaler = fitted_p[-2:]
        #current_scaler = model.seed_scalers[f"{v1,v2}_seed"]
        elapsed_t = perf_counter() - start_t
        for v in (v1,v2):
            offset = model.seed_offsets[f"{v}_seed"]
            scale = model.seed_scalers[f"{v}_seed"]
            logger.info(f"{str(model.tags)}: '{v}'Fitted Offset: {offset}")
            logger.info(f"{str(model.tags)}: '{v}'Fitted Seed Scale: {scale}")

        logger.info(f"{str(model.tags)}: Optimized variant pair '{v1,v2}' in {elapsed_t:0.2f} seconds.")

        model.solve_seir()
        forward_sim_plot(model, outdir,highlight_range=(vwindow_start,vwindow_end))

        if write_specs:
            model.write_specs_to_db()
    return model

def do_variant_optimization(model: RMWCovidModel, outdir:str, variants:list = None, tc_min:float=0.0, tc_max:float=0.99, write_specs=True, **kwargs):
    model.solve_seir()
    model.tags["vopt"] = "pre_opt"
    logger.info(f"{str(model.tags)} Generating pre-optimization plot for comparison...")

    if variants is None:
        variants = model.attrs["variants"]

    logger.info(f"{str(model.tags)} Will optimize variants {variants}.")
    forward_sim_plot(model, outdir)
    for v1,v2 in pairwise(variants):
        if v1 == "none":
            continue
        logger.info(f"{str(model.tags)} Starting optimization process...")
        model.tags["vopt"] = f"{v1}_{v2}"
        start_t = perf_counter()
        # A variant's optimization region start when seeding begins and ends when the variant's proportion of infections
        # is close to 0

        vwindow_start = int(np.clip(
            model.seeds[model.regions[0]][f"{v1}_seed"].argmax() - model.voffset_max - 1, # Give us enough space so that we can optimize the seed backwards up to the limit.
            model.tstart,
            model.tend
        ))
        # The window ends when the variant proportion falls to 0.
        vwindow_end = int(len(model.variant_props) - np.argmax(~np.isclose(model.variant_props[v2].values[::-1],0.0)))
        # Clip the window end so that we do not surpass the end of hospitalizations
        hosps_end_t = model.date_to_t(model.hosps.index.get_level_values("date").max())
        vwindow_end = min(vwindow_end,model.tend,hosps_end_t)
        vwindow_ydict = model.y_dict(vwindow_start) if vwindow_start != 0 else model.y0_dict

        logger.info(f"{str(model.tags)}: Variant pair ({v1}, {v2}) appears between "
                    f"{model.t_to_date(vwindow_start)} and {model.t_to_date(vwindow_end)}")

        # Iterate fitting the variant seeds and fitting TC.
        # opt_result = __optimize_variants(model=model,
        #                                  variants=[v1,v2],
        #                                  tstart=vwindow_start,
        #                                  tend=vwindow_end,
        #                                  yd_start=vwindow_ydict)
        # model.tags["iter"] = f"{i+1}of{n_iters}"
        # model.tags["opt"] = "post_var"
        # forward_sim_plot(model,outdir=outdir)
        # fitted_tc_, fitted_tc_cov = __single_batch_fit(model=model,
        #                                               tc_min=tc_min,
        #                                               tc_max=tc_max,
        #                                               yd_start=vwindow_ydict,
        #                                               tstart=vwindow_start,
        #                                               tend=vwindow_end)
        # model.tags["opt"] = "post_tc"
        # forward_sim_plot(model,outdir=outdir)
        fitted_p, fitted_p_cov = __single_batch_fit_variant_opt(model,
                                                                tc_min=tc_min,
                                                                tc_max=tc_max,
                                                                relevant_variants=[v1, v2],
                                                                yd_start=vwindow_ydict,
                                                                tstart=vwindow_start,
                                                                tend=vwindow_end)
        fitted_tc = fitted_p[:-4]
        #fitted_offset = fitted_p[-4:-2]
        #current_offset = model.seed_offsets[f"{v1,v2}_seed"]
        #fitted_scaler = fitted_p[-2:]
        #current_scaler = model.seed_scalers[f"{v1,v2}_seed"]
        elapsed_t = perf_counter() - start_t
        for v in (v1,v2):
            offset = model.seed_offsets[f"{v}_seed"]
            scale = model.seed_scalers[f"{v}_seed"]
            logger.info(f"{str(model.tags)}: '{v}'Fitted Offset: {offset}")
            logger.info(f"{str(model.tags)}: '{v}'Fitted Seed Scale: {scale}")

        logger.info(f"{str(model.tags)}: Optimized variant pair '{v1,v2}' in {elapsed_t:0.2f} seconds.")

        model.solve_seir()
        forward_sim_plot(model, outdir,highlight_range=(vwindow_start,vwindow_end))

        if write_specs:
            model.write_specs_to_db()
    return model

def do_single_fit_wrapper_parallel(args):
    """Wrapper function for the do_single_fit function that is useful for parallel / multiprocess fitting.

    Two things are necessary here. First, a new logger needs to be created since this wrapper will run in a new process,
    and second, the model must not write results to the database yet. In order to ensure two models aren't given the same
    spec_id, we have to be careful to write to the database serially. So all the models which were fit in parallel will
    be written to the db one at a time after they are all fit.

    Args:
        args: dictionary of arguments for do_single_fit

    Returns: fitted model that is returned by do_single_fit

    """
    setup(os.path.basename(__file__), 'info')
    logger = IndentLogger(logging.getLogger(''), {})
    return do_single_fit(**args, write_results=False)


def do_single_fit_wrapper_nonparallel(args):
    """Wrapper function for the do_single_fit function that is useful for doing multiple fits serially

    This function can be easily mapped to a list of arguments in order to fit several models in succession

    Args:
        args: dictionary of arguments for do_single_fit

    Returns: fitted model that is returned by do_single_fit

    """
    return do_single_fit(**args, write_results=False)


def do_multiple_fits(model_args_list, fit_args, multiprocess = None):
    """Performs multiple model fits, based on a list of model arguments, all using the same fit_args.

    This function can perform fits in parallel or serially, based on the value of multiprocess. This should work cross-
    platform

    Args:
        model_args_list: list of model_args dictionaries, each of which can be used to construct a model
        fit_args: dictionary of fit arguments that will be applied to each model.
        multiprocess: positive integer indicating how many parallel processes to use, or None if fitting should be done serially

    Returns: list of fitted models, order matching the order of models
    """

    # generate list of arguments
    fit_args2 = {key: val for key, val in fit_args.items() if key not in ['write_results', 'write_batch_output']}
    args_list = list(map(lambda x: {**x, **fit_args2}, model_args_list))
    # run each scenario
    if multiprocess:
        #install_mp_handler()  # current bug in multiprocessing-logging prevents this from working right now
        p = Pool(multiprocess)
        models = p.map(do_single_fit_wrapper_parallel, args_list)
    else:
        models = list(map(do_single_fit_wrapper_nonparallel, args_list))

    # write to database serially if specified in those model args
    engine = db_engine()
    if 'write_results' in fit_args and not fit_args['write_results']:
        # default behavior is to write results, so don't write only if specifically told not to.
        pass
    else:
        [m.write_specs_to_db(engine=engine) for m in models]
        #[m.write_results_to_db(engine=engine) for m in models]
        logger.info(f'spec_ids: {",".join([str(m.spec_id) for m in models])}')
        #logger.info(f'result_ids: {",".join([str(m.result_id) for m in models])}')   # takes way too long, let's not write the results to the database right now.

    return models

# region_finder
def do_regions_fit(
                    multiprocess=None,
                   **model_args):
    """Fits a single, disconnected model for each region specified in model_args

    Args:
        model_args: typical model_args used to build a model. Must include list of regions
        fit_args: typical fit_args passed to do_single_fit
        multiprocess: positive int indicating number of parallel processes, or None if fitting should be done serially
    """
    regions = model_args['regions']
    non_region_model_args = {key: val for key, val in model_args.items() if key != 'regions'}
    model_args_list = list(map(lambda x: {'regions': [x], **non_region_model_args, 'tags':{'region': x}}, regions))
    do_multiple_fits(model_args_list, fit_args, multiprocess=multiprocess)

# update_variant
def do_create_report(model, outdir, immun_variants=('ba2121',), from_date=None, to_date=None, prep_model=False, solve_model=False):
    """Create some typically required figures and data for Gov briefings.

    Args:
        model: Model for which to create output
        outdir: path of directory where the output should be saved
        immun_variants: which variants to plot immunity against. Relevant because immune escape factors into immunity and is variant specific
        from_date: start date used in plotting
        to_date: end date used in plotting
        prep_model: whether to run model.prep() before solving the ODEs. useful if model has already been prepped
        solve_model: whether to run model.solve_seir(). useful if model has already been solved.

    Returns: None

    """
    from_date = model.start_date if from_date is None else from_date
    from_date = dt.datetime.strptime(from_date, '%Y-%m-%d').date() if isinstance(from_date, str) else from_date
    to_date = model.end_date if to_date is None else to_date
    to_date = dt.datetime.strptime(to_date, '%Y-%m-%d').date() if isinstance(to_date, str) else to_date

    if prep_model:
        logger.info('Prepping model')
        t0 = perf_counter()
        model.prep(outdir=outdir)
        t1 = perf_counter()
        logger.info(f'Model prepped in {t1 - t0} seconds.')

    if solve_model:
        logger.info('Solving model')
        model.seir_graph_trace()

    subplots_args = {'figsize': (10, 8), 'dpi': 300}

    # prevalence
    fig, ax = plt.subplots(**subplots_args)
    ax.set_ylabel('SARS-CoV-2 Prevalence')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc='best')
    plot_modeled(model, ['I', 'A'], share_of_total=True, ax=ax, label='modeled')
    format_date_axis(ax)
    ax.set_xlim(from_date, to_date)
    ax.axvline(x=dt.date.today(), color='darkgray')
    ax.grid(color='lightgray')
    ax.legend(loc='best')
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'prevalence.png')
    plt.close()

    # hospitalizations
    #TODO: update to be the back_adjusted hosps
    fig, ax = plt.subplots(**subplots_args)
    ax.set_ylabel('Hospitalized with COVID-19')
    plot_observed_hosps(model, ax=ax, color='black')
    plot_modeled(model, 'Ih', ax=ax, label='modeled')
    format_date_axis(ax)
    ax.set_xlim(from_date, to_date)
    ax.axvline(x=dt.date.today(), color='darkgray')
    ax.grid(color='lightgray')
    ax.legend(loc='best')
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'hospitalized.png')
    plt.close()

    # variants
    fig, ax = plt.subplots(**subplots_args)
    plot_modeled(model, ['I', 'A'], groupby='variant', share_of_total=True, ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel('Variant Share of Infections')
    format_date_axis(ax)
    ax.set_xlim(from_date, to_date)
    ax.axvline(x=dt.date.today(), color='darkgray')
    ax.grid(color='lightgray')
    ax.legend(loc='best')
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + 'variant_share.png')
    plt.close()

    # immunity
    for variant in immun_variants:
        fig, ax = plt.subplots(**subplots_args)
        immun = model.immunity(variant=variant)
        immun_65p = model.immunity(variant=variant, age='65+')
        immun_hosp = model.immunity(variant=variant, to_hosp=True)
        immun_hosp_65p = model.immunity(variant=variant, age='65+', to_hosp=True)
        for df, name in zip((immun, immun_65p, immun_hosp, immun_hosp_65p), ('immun', 'immun_65p', 'immun_hosp', 'immun_hosp_65p')):
            df.to_csv(get_filepath_prefix(outdir, tags=model.tags) + f'{name}_{variant}.csv')
        ax.plot(model.daterange, immun, label=f'Immunity vs Infection', color='cyan')
        ax.plot(model.daterange, immun_65p, label=f'Immunity vs Infection (65+ only)', color='darkcyan')
        ax.plot(model.daterange, immun_hosp, label=f'Immunity vs Severe Infection', color='gold')
        ax.plot(model.daterange, immun_hosp_65p, label=f'Immunity vs Severe Infection (65+ only)', color='darkorange')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Percent Immune')
        format_date_axis(ax)
        ax.set_xlim(from_date, to_date)
        ax.axvline(x=dt.date.today(), color='darkgray')
        ax.grid(color='lightgray')
        ax.legend(loc='best')
        fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + f'immunity_{variant}.png')
        plt.close()

    # Immunity and Infections
    colormap_l = [tab20c((4 * i) + j) for i in range(3) for j in range(3)] + \
                 [tab20b((4 * i) + j) for i in range(5) for j in range(3)] + \
                 [tab20c(i + 16) for i in range(3)]

    df = model.solution_ydf.copy()
    df.set_index(pd.date_range(model.start_date, periods=len(df), freq="D"),inplace=True)
    fig, ax = plt.subplots(figsize=(20, 15), nrows=2, sharex=True)
    group_df = df.drop(columns=["D"], level=0).groupby(["variant", "immun"], axis=1).sum()
    group_df = group_df.reindex([x for x in group_df.columns.levels[0] if x != "none"] + ["none"], axis=1,
                                level="variant")
    group_df = group_df.reindex(["high", "medium", "low"], axis=1, level="immun")
    ax[0].stackplot(group_df.index, group_df.T, labels=group_df.columns, colors=colormap_l)
    ax[0].legend(fancybox=False, edgecolor="black", loc="upper left")
    ax[0].set_xlim(group_df.index.min(), group_df.index.max())
    ax[0].set_ylim(0, group_df.sum(axis=1).max())
    ax[0].set_title("Immunity Status by Level and Variant")
    inf_df = df["I"].groupby(["variant", "immun"], axis=1).sum()
    inf_df = inf_df.reindex([x for x in inf_df.columns.levels[0] if x != "none"] + ["none"], axis=1, level="variant")
    inf_df = inf_df.reindex(["high", "medium", "low"], axis=1, level="immun")
    ax[1].stackplot(inf_df.index, inf_df.T, labels=inf_df.columns, colors=colormap_l)
    ax[1].legend(fancybox=False, edgecolor="black", loc="upper left")
    ax[1].set_title("Infections by Variant and Immunity Status")
    fig.tight_layout()
    fig.savefig(get_filepath_prefix(outdir, tags=model.tags) + "immunity_over_time.png")

    do_build_legacy_output_df(model).to_csv(get_filepath_prefix(outdir, tags=model.tags) + 'out2.csv')

    return None


def do_create_report_wrapper_parallel(args):
    """wrapper function for the do_create_report function that can easily be mapped. suitable for creating reports in parallel

    A new logger needs to be created since this wrapper will run in a new process.

    Args:
        args: dictionary of named arguments to be used in do_create_report

    Returns: whatever do_create_report returns

    """
    setup(os.path.basename(__file__), 'info')
    logger = IndentLogger(logging.getLogger(''), {})
    return do_create_report(**args)


def do_create_report_wrapper_nonparallel(args):
    """wrapper function for the do_create_report function that can easily be mapped. suitable for creating reports serially

    Args:
        args: dictionary of named arguments to be used in do_create_report

    Returns: whatever do_create_report returns

    """
    return do_create_report(**args)


def do_create_multiple_reports(models, multiprocess=None, **report_args):
    """Method to easily create multiple reports for various different models. Can be done in parallel or serially

    Args:
        models: list of models that reporst should be created for.
        multiprocess: positive integer indicating how many parallel processes to use, or None if fitting should be done serially
        **report_args: arguments to be passed to do_create_report
    """
    # generate list of arguments
    args_list = list(map(lambda x: {'model': x, **report_args}, models))
    # run each scenario
    if multiprocess:
        #install_mp_handler()  # current bug in multiprocessing-logging prevents this from working right now
        p = Pool(multiprocess)
        p.map(do_create_report_wrapper_parallel, args_list)
    else:
        list(map(do_create_report_wrapper_nonparallel, args_list))


def do_build_legacy_output_df(model: RMWCovidModel):
    """Function to create "legacy output" file, which is a typical need for Gov briefings.

    creates a Pandas DataFrame containing things like prevalence, total infected, and 1-in-X numbers daily for each region

    Args:
        model: Model to create output for.

    Returns: Pandas dataframe containing the output

    """
    totals = model.solution_sum_df(['seir', 'region']).stack(level=1)
    totals['region_pop'] = totals.sum(axis=1)
    totals = totals.rename(columns={'Ih': 'Iht', 'D': 'Dt', 'E': 'Etotal'})
    totals['Itotal'] = totals['I'] + totals['A']

    age_totals = model.solution_sum_df(['seir', 'age'])
    age_totals = age_totals.drop(columns=['A', 'E', 'S', 'I'])

    age_df = pd.DataFrame()
    # agecat_finder
    age_df['D_age1'] = age_totals['D']['0-17']
    age_df['D_age2'] = age_totals['D']['18-64']
    age_df['D_age3'] = age_totals['D']['65+']

    age_df['Ih_age1'] = age_totals['Ih']['0-17']
    age_df['Ih_age2'] = age_totals['Ih']['18-64']
    age_df['Ih_age3'] = age_totals['Ih']['65+']

    df = totals.join(model.new_infections).join(model.new_infections_symptomatic).join(model.re_estimates).join(age_df)

    df['prev'] = 100000.0 * df['Itotal'] / df['region_pop']
    df['oneinX'] = df['region_pop'] / df['Itotal']

    return df


def do_fit_scenarios(base_model_args, scenario_args_list, fit_args, multiprocess=None):
    """Fits several models using a base set of arguments and a list of scenarios which apply changes to the base settings

    Args:
        base_model_args: dictionary of model args that are common to all scenarios being fit
        scenario_args_list: list of dictionaries, each of which modifies the base_model_args for a particular scenario
        fit_args: fitting arguments applied to all scenarios
        multiprocess: positive integer indicating how many parallel processes to use, or None if fitting should be done serially

    Returns:

    """
    # construct model args from base model args and scenario args list
    model_args_list = []
    for scenario_args in scenario_args_list:
        model_args_list.append(copy.deepcopy(base_model_args))
        model_args_list[-1].update(scenario_args)

    return do_multiple_fits(model_args_list, fit_args, multiprocess)

def do_create_immunity_decay_curves(model, cmpt_attrs):
    print("test")
