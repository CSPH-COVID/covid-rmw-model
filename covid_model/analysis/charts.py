""" Python Standard Library """
import os
from time import perf_counter
import datetime as dt
from os.path import join
from itertools import product
""" Third Party Imports """
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.cm import tab10, tab20, tab20b, tab20c

""" Local Imports """
from covid_model.model import CovidModel
from covid_model.utils import db_engine
from covid_model.data_imports import ExternalHospsEMR, ExternalHospsCOPHS


def plot_observed_hosps(model, ax, **plot_params):
    """ Plots actual/observed (from data source) hospitalizations vs model's estimated hospitalizations (modeled) over
        time.
    :param model: The instance of the model to retrieve actual hospitalizations from.
    :param ax: The axes to draw the hospitalizations onto.
    :param plot_params: Any additional plotting parameters, passed directly to Axes.plot().
    :return: None
    """
    for name, group in model.hosps.groupby("region"):
        ax.plot(group.reset_index("region", drop=True), label=f"Actual Hosps. ({name})", **plot_params)


def plot_modeled_vs_actual_hosps():
    # TODO
    pass


def plot_modeled(model, compartments, ax=None, transform=lambda x: x, groupby=[], share_of_total=False, from_date=None,
                 **plot_params):
    if type(compartments) == str:
        compartments = [compartments]

    if groupby:
        if type(groupby) == str:
            groupby = [groupby]
        df = transform(model.solution_sum_df(['seir', *groupby])[compartments].groupby(groupby, axis=1).sum())
        if share_of_total:
            total = df.sum(axis=1)
            df = df.apply(lambda s: s / total)
    else:
        df = transform(model.solution_sum_df('seir'))
        if share_of_total:
            total = df.sum(axis=1)
            df = df.apply(lambda s: s / total)
        df = df[compartments].sum(axis=1)

    if from_date is not None:
        df = df.loc[from_date]

    df.plot(ax=ax, **plot_params)

    if share_of_total:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def plot_modeled_by_group(model, axs, compartment='Ih', **plot_params):
    for g, ax in zip(model.groups, axs.flat):
        ax.plot(model.daterange, model.solution_ydf.xs(g, level='group')[compartment],
                **{'c': 'blue', 'label': 'Modeled', **plot_params})
        ax.set_title(g)
        ax.legend(loc='best')
        ax.set_xlabel('')


def plot_transmission_control(model, ax, regions=None):
    use_variants = sorted(list(set(model.attrs["variant"]) - {"none"}))
    color_idx = {v: (2 * iv) for iv, v in enumerate(use_variants)}
    # Use all regions by default, but can be overwritten if passed as argument.
    regions = model.regions if regions is None else regions
    # need to extend one more time period to see the last step. Assume it's the same gap as the second to last step
    tc_df = pd.DataFrame.from_dict(model.tc, orient='index').set_index(
        np.array([model.t_to_date(t) for t in model.tc.keys()]))
    # Subset to just the regions we want to plot
    tc_df = tc_df[regions]  # regions should be a list
    # Get the scaling values for each variant in each region (assumes that betta *only* varies between variants and
    # regions, and not other attributes).
    beta_scale = model.get_param_for_attrs_by_t(param="betta",
                                                attrs=dict(variant=[x for x in model.attrs["variant"] if x != "none"],
                                                           region=[x for x in regions]))\
        .unstack(level=["region","variant"])\
        .droplevel(["age","immun","vacc"],axis=0)\
        .droplevel(level=0,axis=1)\
        .drop_duplicates()
    # If a level (i.e. compartment level like age, variant, etc) has a value that never changes, we don't need to
    # plot that level.
    # special_lvls = {"t","region"}
    # lvls_to_drop = {name for name in beta_scale.index.names if beta_scale.groupby(name).ngroups == 1 and name not in special_lvls} - {"region"}
    # unstack_lvls = ["region"] + list(set(beta_scale.index.names) - lvls_to_drop - special_lvls)
    # beta_scale = beta_scale.droplevel(level=list(lvls_to_drop),axis=0).unstack(unstack_lvls)
    beta_scale.index = pd.Index([model.t_to_date(t) for t in beta_scale.index],name="date")
    beta_scale = beta_scale.reindex(pd.date_range(model.start_date, model.end_date)).ffill()
    #
    # Multiply by beta (tc) df so that we get the effective beta at each timestep.
    beta_scaled_t = tc_df.reindex(pd.date_range(model.start_date,model.end_date)).multiply(beta_scale,axis=1,level="region").dropna()
    # Compute a mask for each variant, so we don't plot the effective beta for variants we don't care about anymore
    var_starts = {v: pd.to_datetime(model.t_to_date(int(np.argmax(np.array(model.seeds[regions[0]][f"{v}_seed"]))))) for v in use_variants}
    sol_var_props_r = np.flip(model.solution_var_props(model.tstart, model.tend, use_variants).reshape(len(use_variants),-1),axis=-1)
    v_ends = sol_var_props_r.shape[1] - np.argmax(sol_var_props_r > 0, axis=-1)
    var_ends = {v: pd.to_datetime(model.t_to_date(int(v_ends[i]))) for i,v in enumerate(use_variants)}
    for region in regions:
        ax.step(tc_df.index, tc_df[region], where="post", label=f"Base Beta ({region})", color="black", linestyle="--", zorder=999)
        for variant in use_variants:
            masked_beta_t = beta_scaled_t.loc[(beta_scaled_t.index >= var_starts[variant]) & (beta_scaled_t.index <= var_ends[variant]), (region, variant)]
            if len(masked_beta_t) != 0:
                ax.step(masked_beta_t.index,
                        masked_beta_t,
                        where="post",
                        label=f"Scaled Beta ({region}, {variant})",
                        color=tab20(color_idx[variant]))
    ax.legend(fancybox=False, edgecolor="black")
    #tc_df.plot(drawstyle="steps-post", xlim=(model.start_date, model.end_date), **plot_params)

def plot_beta_r(model, ax):
    # TODO fix this
    fig, ax = plt.subplots(figsize=(24, 40), nrows=5, sharex=True)

    # for c in effective_beta.columns:
    #     plot_eff_beta = effective_beta.loc[(effective_beta.index >= first_appear[c]) & (effective_beta.index <= last_appear[c]),c]
    #     ax[0].plot(plot_eff_beta.index, plot_eff_beta, label=c,)
    # ax[0].scatter(plot_eff_beta.index, plot_eff_beta)
    ax[0].set_title("$\\beta_{base}$ (Fitted Parameter) over Time")
    ax[0].step(beta.index, beta["beta"], color="black", label="Base Beta", where="post")
    ax[0].set_ylabel("Base Beta ($\\beta_{base}$)")

    ax[1].set_title("$\\beta_{mult}$ (Variant Multiplier * Lambda Weighted Avg.) over Time")
    for i, (_, c) in enumerate(effective_beta_mult.columns):
        relevant_region = (effective_beta_mult.index >= first_appear[c]) & (effective_beta_mult.index <= last_appear[c])
        plot_efb = effective_beta_mult[("beta", c)]
        ax[1].plot(plot_efb.index, plot_efb, color=tab10(i), linestyle="--", alpha=0.4)
        ax[1].plot(plot_efb.index, plot_efb.mask(~relevant_region), color=tab10(i), linewidth=2,
                   label=f"$\\beta_{{mult}}$ ({c})")
        # ax[1].plot(beta_mults.index,beta_mults[c],color=tab10(i),alpha=0.4,linestyle="--")
        # ax[1].plot(beta_mults.index,3.125* beta_mults[c],color=tab10(i),alpha=0.4,linestyle="--")
    ax[1].legend(fancybox=False, edgecolor="black")
    ax[1].set_ylabel("Beta Multiplier ($\\beta_{mult}$)")

    ax[2].set_title("Variant Proportions over Time")
    for i, c in enumerate(inf_share_per_variant.columns):
        ax[2].plot(inf_share_per_variant.index, inf_share_per_variant[c], color=tab10(i), label=c, linewidth=2)
    ax[2].legend(fancybox=False, edgecolor="black")
    ax[2].set_ylabel("Variant Proportions")

    ax[3].plot(weighted_effective_beta.index, weighted_effective_beta, linewidth=2, color="black",
               label="Weighted $\\beta_{e}$")
    ax[3].stackplot(effective_beta_weights.index, effective_beta_weights.T,
                    labels=[f"{c} Contribution" for c in effective_beta_weights.columns], alpha=0.4)
    ax[3].legend(fancybox=False, edgecolor="black")
    ax[3].set_ylabel("Beta")
    ax[3].set_title("$\\beta_e$ (Effective Beta) with Weighted Average")
    ax[3].set_ylabel("Effective Beta ($\\beta_{e}$)")

    ax[4].plot(weighted_effective_beta.index, weighted_effective_beta / gamm, linewidth=2, color="black",
               label="Weighted $R_{e}$")
    ax[4].stackplot(effective_beta_weights.index, effective_beta_weights.T / gamm,
                    labels=[f"{c} Contribution" for c in effective_beta_weights.columns], alpha=0.4)
    ax[4].axhline(1.0, linestyle="--", alpha=0.5, color="black")
    ax[4].set_yticks([0, 1, 2, 4, 6, 8, 10])
    ax[4].set_ylabel("Effective R ($R_{e}$)")
    ax[4].legend(fancybox=False, edgecolor="black")
    ax[4].set_title("$R_e$ (Effective R) with Weighted Average")

    fig.tight_layout()

def plot_variant_proportions(model, ax, show_seeds=None):
    # Plot variant proportions
    use_variants = sorted(list(set(model.attrs["variant"]) - {"none"}))
    color_idx = {v: (2 * iv) for iv, v in enumerate(use_variants)}
    actual_var = model.variant_props.drop(columns=["none"])
    for vname, vprop in zip(actual_var.columns, actual_var.T.values):
        ax.plot(np.array([model.t_to_date(t) for t in actual_var.index]),
                vprop,
                color=tab20(color_idx[vname] + 1),
                linestyle="--")
    obs_variants = model.solution_var_props(tstart=model.tstart, tend=model.tend, variants=use_variants).reshape(
        len(use_variants), -1)
    for vname, vprop in zip(use_variants, obs_variants):
        ax.plot(np.array([model.t_to_date(t) for t in range(model.tstart, model.tend + 1)]),
                vprop,
                label=f"{vname}",
                color=tab20(color_idx[vname]))
    ax.legend(fancybox=False, edgecolor="black", loc="lower left")
    if show_seeds is not None:
        for variant in show_seeds:
            seed_p = min(key for key in model.seeds[model.regions[0]][f"{variant}_seed"] if key != 0)
            seed_offset = model.seed_offsets[model.regions[0]][f"{variant}_seed"]
            effective_seed = max(0, min(model.tend, seed_offset+seed_p))
            ax.axvline(x=model.t_to_date(seed_p),color=tab20(color_idx[variant]))
            ax.axvline(x=model.t_to_date(effective_seed),color=tab20(color_idx[variant]+1))

def format_date_axis(ax, interval_months=None, **locator_params):
    locator = mdates.MonthLocator(interval=interval_months) if interval_months is not None else mdates.AutoDateLocator(
        **locator_params)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(None)


def draw_stackplot(ax, df, f_title, l_title, ylabel, xlabel, colors):
    # Plot the dataframe values on the axis
    ax.stackplot(df.columns, df.values, labels=df.index, colors=colors)
    # Set up legend.
    ax.legend(title=l_title, fancybox=False, edgecolor="black", bbox_to_anchor=(1.0, 1.01), loc="upper left",
              fontsize=14, title_fontsize=14)
    # Set X-axis limit and label
    ax.set_xlim(df.columns.min(), df.columns.max())
    ax.set_xlabel(xlabel, fontsize=14)
    # Set Y-axis limit and label
    ax.set_ylabel(ylabel, fontsize=14)
    # Set title.
    ax.set_title(f_title, fontsize=16)


def plot_seir_comparments(df, fig_title, fig_filename=None, figsize=(14, 9)):
    seir = df.groupby(level="seir", axis=1).sum().T
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=seir,
                   f_title=fig_title,
                   l_title="Compartments",
                   xlabel="Time",
                   ylabel="Population",
                   colors=[tab10(i) for i in range(seir.index.nunique())])
    ax.set_ylim(0, seir.sum().max())
    plt.tight_layout()
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def plot_vacc_status_props(df, fig_title, fig_filename=None, figsize=(14, 9)):
    vacc_status = df.groupby(level=["vacc", "age"], axis=1).sum().T
    # We use gray for the none vaccination status, and colors otherwise
    colors = []
    color_groups = np.array([0, 1, 4, 2, 3])
    for vacc, cgrp in zip(vacc_status.index.get_level_values("vacc").unique(), color_groups):
        for j, age in enumerate(vacc_status.index.get_level_values("age").unique()):
            colors.append(tab20c((cgrp * 4) + j))
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=vacc_status,
                   f_title=fig_title,
                   l_title="Compartments",
                   xlabel="Time",
                   ylabel="Population",
                   colors=colors)
    ax.set_ylim(0, vacc_status.sum().max())
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def plot_variant_props(df, fig_title, fig_filename=None, figsize=(14, 9)):
    variants = df.drop(axis=1, level="seir", labels=["S", "E"]).groupby(level=["variant"], axis=1).sum().T
    variants = variants.divide(variants.sum(axis=0))
    variants.loc["none", variants.loc["none", :].isna()] = 1.0
    variants.fillna(0.0, inplace=True)
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=variants,
                   f_title=fig_title,
                   l_title="Compartments",
                   ylabel="Normalized Variant Proportion",
                   xlabel="Time",
                   colors=[tab10(i) for i in range(variants.index.nunique())])
    ax.set_ylim(0, 1)
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def plot_immunity_props(df, fig_title, fig_filename=None, figsize=(14, 9)):
    immun_df = df.groupby(level=["age", "immun"], axis=1).sum().T
    # reorder so that the colors make more intuitive sense.
    immun_df = immun_df.reindex(index=["strong", "weak", "none"], level=1)
    colors = []
    for i, age in enumerate(immun_df.index.get_level_values("age").unique()):
        for j, immun in enumerate(immun_df.index.get_level_values("immun").unique()):
            colors.append(tab20c((4 * i) + j))
    fig, ax = plt.subplots(figsize=figsize)
    draw_stackplot(ax=ax,
                   df=immun_df,
                   f_title=fig_title,
                   l_title="Compartments",
                   xlabel="Time",
                   ylabel="Population",
                   colors=colors)
    ax.set_ylim(0, immun_df.sum().max())
    if fig_filename is not None:
        plt.savefig(fig_filename)
    return fig, ax


def generate_stackplots(model, output_dir=None, start_date=None, end_date=None):
    # Make a copy of the solution DF
    solution_df = model.solution_ydf.copy()

    # Set index to the actual dates for better readability
    solution_df.set_index(pd.date_range(model.start_date, periods=len(solution_df)), inplace=True)

    # Start and end dates can be specified, otherwise we use the full range of the model's solution
    start_date = model.start_date if start_date is None else start_date
    end_date = model.end_date if end_date is None else end_date
    solution_df = solution_df.loc[start_date:end_date]

    # Output directory defaults to current directory if not specified
    output_dir = os.getcwd() if output_dir is None else output_dir

    # Results figures and axes
    result_figs = {}

    # Plot SEIR
    seir_fig, seir_ax = plot_seir_comparments(df=solution_df,
                                              fig_title="SEIR Compartments",
                                              fig_filename=join(output_dir, "seir_status.png"))
    result_figs["seir"] = (seir_fig, seir_ax)

    # Plot Vaccination Status
    vacc_fig, vacc_ax = plot_vacc_status_props(df=solution_df,
                                               fig_title="Vaccination Status",
                                               fig_filename=join(output_dir, "vacc_status.png"))
    result_figs["vacc"] = (vacc_fig, vacc_ax)

    # Plot Variant Proportions
    var_fig, var_ax = plot_variant_props(df=solution_df,
                                         fig_title="Variant Distribution",
                                         fig_filename=join(output_dir, "variant_proportions.png"))
    result_figs["var"] = (var_fig, var_ax)

    # Plot Immunity Proportions
    immun_fig, immun_ax = plot_immunity_props(df=solution_df,
                                              fig_title="Immunity Status/Age",
                                              fig_filename=join(output_dir, "immunity_status.png"))
    result_figs["immun"] = (immun_fig, immun_ax)

    # Return results figures
    return result_figs
