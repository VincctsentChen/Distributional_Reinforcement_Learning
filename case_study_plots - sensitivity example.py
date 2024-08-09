# =======================================================
# Plots of Results - Hypertension treatment case study
# =======================================================

# Loading modules
import os  # directory changes
import numpy as np  # matrix operations
import pandas as pd  # data frame operations
import matplotlib # plotting configurations
# matplotlib.use("Agg") # making sure interactive mode is off
import matplotlib.pyplot as plt #base plots
# plt.ioff() # making sure interactive mode is off
import seaborn as sns #plots

# Plotting parameters
sns.set_style("ticks")
paper_rc = {'lines.linewidth': 1, 'lines.markersize': 6, 'font.size': 14}
sns.set_context("paper", rc=paper_rc)

# Function to plot demographic summary by BP category
def plot_demo_bp(demo):

    # Figure parameters
    n_colors = 6 # number of colors in palette
    xlims = [-0.5, 3.5] # limit of the x-axis
    ylims = [0, 4] # limits of the y-axis
    y_ticks = np.arange(6)
    data_labe_size = 8 # font size for data labels
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    cat_labels = demo.bp_cat.unique()  # BP categories

    # Making plot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    g1 = sns.barplot(x="bp_cat", y="wt", hue="sex", data=demo[demo.race == 0], palette=sns.color_palette("Greys", n_colors)[3:-1], ax=axes[0])
    g2 = sns.barplot(x="bp_cat", y="wt", hue="sex", data=demo[demo.race == 1], palette=sns.color_palette("Greys", n_colors)[3:-1], ax=axes[1])

    # Adding data labels
    for p in g1.patches:
        g1.annotate(format(p.get_height(), '.2f'), (p.get_x()+p.get_width()/2., p.get_height()), ha='center',
                       va='center', xytext=(0, 10), textcoords='offset points', fontsize=data_labe_size)

    for p in g2.patches:
        g2.annotate(format(p.get_height(), '.2f'), (p.get_x()+p.get_width()/2., p.get_height()), ha='center',
                       va='center', xytext=(0, 10), textcoords='offset points', fontsize=data_labe_size)

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('BP Catgory', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Number of People (in Millions)', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.12)

    ## Adding subtitles
    axes[0].set_title('Race: Black', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Race: White', fontsize=subtitle_size, fontweight='semibold')

    for ax in fig.axes:
        plt.sca(ax)
        # plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(risk_labels)), labels=risk_labels, ha='center')
        plt.xticks(fontsize=tick_size-1, ticks=np.arange(len(cat_labels)), labels=cat_labels,
                   ha='center')  # rotation=15,
        plt.yticks(fontsize=tick_size)

    # Modifying Legend
    handles, _ = axes[0].get_legend_handles_labels()
    labels = ['Females', 'Males']
    axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(1.1, -0.12),
                      handles=handles, labels=labels, prop={'size': legend_size})
    axes[1].get_legend().remove()

    fig.set_size_inches(6.5, 3.66)
    plt.savefig("demographics_bp_category.eps", bbox_inches='tight')
    plt.close()

# Function to plot distribution of treatment by BP category at year 1 and 10
def plot_trt_dist(trt_df):

    # Figure parameters
    xlims = [-0.5, 3.5] # limit of the x-axis
    ylims = [-0.5, 5.5] # limits of the y-axis
    y_ticks = np.arange(6)
    axes_size = 10 # font size of axes labels
    subtitle_size = 9 # font size for subplot titles
    tick_size = 8 # font size for tick labels
    legend_size = 8 # font size for legend labels
    flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markeredgewidth=0.5, markersize=2,
                      linestyle='none') # outliers properties

    # Making plots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==1)],
                palette="viridis", linewidth=0.75, flierprops=flierprops, ax=axes[0])
    sns.boxplot(x="bp_cat", y="meds", hue="policy", data=trt_df[(trt_df.year==10)],
                palette="viridis", linewidth=0.75, flierprops=flierprops, ax=axes[1])

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('BP Category', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='',
             ylim=ylims, yticks=y_ticks, ylabel='')
    fig.subplots_adjust(bottom=0.3, wspace=0.2, hspace=0.45)

    # Adding subtitles
    bp_labels = ['Normal BP', 'Elevated BP', 'Stage 1\nHypertension', 'Stage 2\nHypertension']
    axes[0].set_title('Year 0', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Year 9', fontsize=subtitle_size, fontweight='semibold')
    fig.subplots_adjust(bottom=0.3)  # for overal plots

    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=15, fontsize=tick_size-1, ticks=np.arange(len(bp_labels)), labels=bp_labels, ha='center') # for overall plots
        plt.yticks(fontsize=tick_size)

    # Modifying Legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(1.08, -0.4),
                      handles=handles, labels=labels, prop={'size': legend_size})
    axes[1].get_legend().remove()

    fig.set_size_inches(6.5, 3)
    plt.savefig("treatment_per_policy.eps", bbox_inches='tight')
    plt.close()

# Function to plot frequency of drug type per number of medication
def freq_drug(df):
    # Figure parameters
    ylims = [0, 15]  # limits of the y-axis
    y_ticks = np.arange(0, 15.1, 5) # ticks in the y-axis
    axes_size = 9  # font size of axes labels
    subtitle_size = 9  # font size for subplot titles
    title_size = 11 # font size for titles
    tick_size = 7  # font size for tick labels
    legend_size = 8  # font size for legend labels
    line_width = 0.8  # width for lines in plots
    bp_cat_labels = ['Elevated BP', 'Stage 1 Hypertension', 'Stage 2 Hypertension']
    year_labels = ['Year 0\n', 'Year 9\n']

    # Barplots by BP level per Year
    for y, year in enumerate(year_labels):
        # Barplot by BP level
        fig, axes = plt.subplots(nrows=1, ncols=3)
        for p, bp in enumerate(bp_cat_labels):
            ## Making plot
            sns.barplot(x="drug_type", y="freq", hue="policy", data=df[(df.year == df.year.unique()[y]) & (df.bp_cat == bp)],
                        palette="viridis", dodge=0.25, ci=None, ax=axes[p])
            ## Adding subtitles
            axes[p].set_title(bp, fontsize=subtitle_size, fontweight='semibold')

        # Figure Configuration
        ## Configuration for the panel plot
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.title(year, fontsize=title_size, fontweight='semibold')
        plt.ylabel('Frequency (in millions)\n\n', fontsize=axes_size, fontweight='semibold')
        if y==1: # year 9
            plt.xlabel('\nMedication Type', fontsize=axes_size, fontweight='semibold')

        ## Axes configuration for individual subplots
        plt.setp(axes, xlabel='', ylabel='', ylim=ylims, yticks=y_ticks)
        for a, ax in enumerate(fig.axes):
            ax.set_yticklabels(['{0:,.0f}'.format(x) for x in ax.get_yticks()])
            plt.sca(ax)
            plt.xticks(fontsize=tick_size)
            if a > 0:
                plt.yticks([])
            else:
                plt.yticks(fontsize=tick_size)

        # Modifying Legend
        fig.subplots_adjust(bottom=0.3)
        if y == 1: # year 9
            handles, labels = axes[0].get_legend_handles_labels()
            axes[0].legend(loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(1.7, -0.4),
                           handles=handles, labels=labels, prop={'size': legend_size})
            axes[1].get_legend().remove()
            axes[2].get_legend().remove()
        else: # year 0
            axes[0].get_legend().remove(); axes[1].get_legend().remove(); axes[2].get_legend().remove()

        ## Saving plot
        fig.set_size_inches(8, 2.5)
        plt.savefig("frequency_medication_type_year" + year_labels[y].split()[1] + ".eps", bbox_inches='tight')
        plt.close()

# Function to plot number of QALYs saved or events prevented per capita by BP category
def qalys_events(df, events=False):
    # Figure parameters
    axes_size = 14  # font size of axes labels
    legend_size = 10  # font size for legend labels
    tick_size = 12  # font size for tick labels
    line_width = 0.8  # width for lines in plots

    # Making plot
    ## Pointplot by BP level
    fig, ax = plt.subplots()
    if events:
        sns.barplot(x="bp_cat", y="events", hue="policy", data=df, estimator=np.sum,
                    palette="viridis", errwidth=line_width, dodge=0.25, ci=95, n_boot=10000, seed=1)
    else:
        sns.barplot(x="bp_cat", y="qalys", hue="policy", data=df, estimator=np.sum,
                    palette="viridis", errwidth=line_width, dodge=0.25, ci=95, n_boot=10000, seed=1)

    ## Figure Configuration
    ### Configuration for the plot
    plt.xlabel('BP Category', fontsize=axes_size, fontweight='semibold')
    if events:
        plt.ylabel('ASCVD Events Averted\nper 100,000 Patients', fontsize=axes_size, fontweight='semibold')
    else:
        plt.ylabel('QALYs Saved\nper 100,000 Patients', fontsize=axes_size, fontweight='semibold')
    plt.xticks(fontsize=tick_size-2)
    plt.yticks(fontsize=tick_size)

    ### Formatting x-axis
    ax.set_xticklabels(['Elevated BP', 'Stage 1\nHypertension', 'Stage 2\nHypertension'])  # 'Normal BP',

    ### Formatting y-axis (adding commas) - gives a warning but works well (for totals only)
    ax.set_yticklabels(['{0:,.0f}'.format(x) for x in ax.get_yticks()])

    ### Legend configuration
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.25),
               handles=handles, labels=labels, prop={'size': legend_size})

    ## Saving plot
    fig.set_size_inches(4, 3)
    if events:
        plt.savefig("events_prevented.eps", bbox_inches='tight')
    else:
        plt.savefig("qalys_saved.eps", bbox_inches='tight') #, dpi=300

# Function to plot price of interpretability per capita (or total)
def price_interpret(pi_df, total=False):

    # Figure parameters
    # ylims = [0, 2.25]  # limits of the y-axis
    # y_ticks = np.arange(0, 2.25, 0.5)
    axes_size = 9 # font size of axes labels
    legend_size = 7  # font size for legend labels
    tick_size = 8  # font size for tick labels
    line_width = 0.8  # width for lines in plots

    # Making plot
    ## Pointplot by BP level
    fig, ax = plt.subplots()
    # sns.pointplot(x="bp_cat", y="pi", hue="policy", data=pi_df, estimator=np.sum,
    #               palette="viridis", join=False, errwidth=line_width, dodge=0.25, ci=90, n_boot=10000, seed=1) # point plot (totals)
    sns.barplot(x="bp_cat", y="pi", hue="policy", data=pi_df, estimator=np.sum,
                palette="viridis", errwidth=line_width, dodge=0.25, ci=95, n_boot=10000, seed=1) # point plot (totals)

    ## Figure Configuration
    ### Configuration for the plot
    plt.xlabel('BP Category', fontsize=axes_size, fontweight='semibold')
    if total:
        plt.ylabel('Total Price of Interpretability', fontsize=axes_size, fontweight='semibold')
    else:
        plt.ylabel('QALYs Saved\nper 100,000 Patients', fontsize=axes_size, fontweight='semibold') # 'Price of Interpretability\nper 100,000 Patients'
    plt.xticks(fontsize=tick_size-1)
    plt.yticks(fontsize=tick_size) #ticks=y_ticks,

    ### Formatting x-axis
    ax.set_xticklabels(['Elevated BP', 'Stage 1\nHypertension', 'Stage 2\nHypertension']) # 'Normal BP',

    ### Formatting y-axis (adding commas) - gives a warning but works well
    if total:
        ax.set_yticklabels(['{0:,.0f}'.format(x) for x in ax.get_yticks()])

    ### Legend configuration
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.25),
                   handles=handles, labels=labels, prop={'size': legend_size})

    ## Saving plot
    fig.set_size_inches(4, 3)
    if total:
        plt.savefig("toptal_pi.eps", bbox_inches='tight')
    else:
        plt.savefig("pi_per_capita.eps", bbox_inches='tight') #, dpi=300

# Function to the distribution of the price of interpretability
def price_interpret_dist(pi_df, pairwise=False):

    # Figure parameters
    subtitle_size = 7.5  # font size for subplot titles
    axes_size = 7.5  # font size of axes labels
    legend_size = 8 # font size of legend
    tick_size = 7  # font size for tick labels
    mks = ['o', 'X']
    marker_size = 15 # marker size

    # Making figure
    if pairwise: # pairwise differences between CMP and MP
        g = sns.lmplot(data=pi_df, x="risk10", y="pi", col="bp_cat", scatter_kws={'s': marker_size, 'alpha': 0.5, 'color': 'gray'},
                       line_kws={'color': 'gray'}, fit_reg=True, order=2, truncate=True, legend=False, sharey=False) #palette='Greys',
    else: # PI (differences between CMP and MP with the optimal policy)
        g = sns.lmplot(data=pi_df, x="risk10", y="pi", hue="policy", col="bp_cat", scatter_kws={'s': marker_size},
                       markers=mks, palette='viridis', fit_reg=True, order=2, truncate=True, legend=False, sharey=False)

    ## Figure Configuration
    sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    g.set_titles(col_template="{col_name}", fontweight='semibold') #fontsize=subtitle_size,
    plt.subplots_adjust(wspace=0.3)
    g.set_axis_labels("", "Total Discounted Reward Difference\n(in QALYs)", fontsize=axes_size, fontweight='semibold')  # 10-Year Risk for ASCVD Events # Mean Arterial Pressure # SBP
    if pairwise:
        g.set(xlim=(-0.01, 0.61), xticks=np.arange(0, 0.61, 0.1), ylim=(-0.0015, 0.051), yticks=np.arange(0, 0.051, 0.01)) #
        g.set_xticklabels(np.arange(0, 0.61, 0.1).round(2), fontsize=tick_size)
        g.set_yticklabels(np.arange(0, 0.051, 0.01).round(2), fontsize=tick_size)
        g.axes.flat[1].text(0.7, -0.015, '10-Year Risk for ASCVD Events', fontsize=axes_size, fontweight='semibold',
                            horizontalalignment='center')
    else:
        g.set(xlim=(-0.01, 0.61), xticks=np.arange(0, 0.61, 0.1), ylim=(-0.005, 0.205), yticks=np.arange(0, 0.21, 0.05)) # xlim=(-0.001, 0.61), xticks=np.arange(0, 0.61, 0.1),
        g.set_xticklabels(np.arange(0, 0.61, 0.1).round(2), fontsize=tick_size)
        g.set_yticklabels(np.arange(0, 0.21, 0.05).round(2), fontsize=tick_size)
        g.axes.flat[1].text(0.7, -0.06, '10-Year Risk for ASCVD Events', fontsize=axes_size, fontweight='semibold',
                            horizontalalignment='center')
    g.set_xticklabels(fontsize=tick_size)

    ## Saving plot
    g.fig.set_size_inches(6.5, 2)
    if pairwise: # pairwise differences between CMP and MP
        plt.savefig("scatterplot_pairwise_differences.jpeg", bbox_inches='tight', dpi=600)
    else: # PI (differences between CMP and MP with the optimal policy)
        ### Legend configuration
        plt.legend(loc='upper center', ncol=2, frameon=False, prop={'size': legend_size}, bbox_to_anchor=(-1.5, -0.3))
        plt.savefig("scatterplot_pi.jpeg", bbox_inches='tight', dpi=600)

# Function to plot the policies of a single patient over states for a single year
def plot_policies_state_pt(policy_df, pt_dat):

    # Figure parameters
    xlims = [-0.5, 5.5]  # limit of the x-axis
    ylims = [-0.5, 5.5]  # limits of the y-axis
    axes_size = 20  # font size of axes labels
    subtitle_size = 24  # font size for subplot titles
    tick_size = 20  # font size for tick labels
    legend_size = 20  # font size for legend labels
    data_label_size = 20  # font size of data labels
    marker_size = 60 # marker size

    # Making figure
    fig, axes = plt.subplots()
    mks = ['X', 'P', 'o']
    sns.scatterplot(x='state', y='meds_jitt', data=policy_df,
                    hue=policy_df.trt.astype('category').cat.codes,
                    hue_order=np.arange(policy_df.trt.unique().shape[0]),
                    style=policy_df.policy.astype('category').cat.codes,
                    markers=mks, s=marker_size,
                    palette=sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex())

    # Adding vertical lines to identify state classes in plots
    axes.vlines(x=[0.5, 2.5, 4.5], ymin=ylims[0], ymax=ylims[1], color='darkgray', linestyle='dotted')

    # Adding horizontal lines to identify action classes in plots
    axes.hlines(y=[0.5, 1.5, 2.5, 3.5, 4.5], xmin=xlims[0], xmax=xlims[1], color='darkgray', linestyle='dotted')

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('\nHealth Condition', fontsize=axes_size, fontweight='semibold')
    plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='', ylim=ylims, yticks=range(6), ylabel='')
    fig.subplots_adjust(bottom=0.2, right=0.85)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45, fontsize=tick_size-1)
        plt.yticks(fontsize=tick_size)

    # Modifying Legends
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    ## Marker legend
    handles = [f(m, 'black') for m in mks]
    labels = ['CMP', 'MP', 'OP']
    axes.legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2),
                handles=handles, labels=labels, prop={'size': legend_size})

    # Adding data labels
    colors = sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex()
    labels = policy_df.trt.unique()[policy_df.trt.astype('category').cat.codes.unique().argsort()]
    for row in range(policy_df.shape[0]):
        axes.text(policy_df.state_id[row], policy_df.meds_jitt[row]+0.2, policy_df.trt[row],
                  horizontalalignment='center', fontsize=data_label_size, color=colors[int(np.where(policy_df.trt[row]==labels)[0])], zorder=0)

    # Adding title
    axes.set_title(pt_dat, fontsize=subtitle_size, fontweight='semibold')

    # Saving plot
    fig.set_size_inches(11, 8.5)
    plt.savefig('Patient '+str(policy_df.profile[0])+'.png', dpi=100, bbox_inches='tight') #' - '+pt_dat.translate(str.maketrans('', '', '\n'))+
    plt.close()

# Function to plot the policies of selected patients over states for a single year
def plot_policies_state(policy_df):

    # Figure parameters
    xlims = [-0.5, 5.5] # limit of the x-axis
    ylims = [-0.5, 1.5] # limits of the y-axis # [-0.5, 5.5]
    axes_size = 7.5 # font size of axes labels
    subtitle_size = 7.5 # font size for subplot titles
    tick_size = 7 # font size for tick labels
    legend_size = 6.5 # font size for legend labels
    data_label_size = 5.5 # font size of data labels
    marker_size = 10 # marker size

    # Making figure
    fig, axes = plt.subplots(nrows=1, ncols=4)
    mks = ['^', 'o', 's']
    sns.scatterplot(x='state', y='meds_jitt', data=policy_df[policy_df['profile']==policy_df['profile'].unique()[0]],
                    hue=policy_df.trt.astype('category').cat.codes,
                    hue_order=np.arange(policy_df.trt.unique().shape[0]), style=policy_df.policy.astype('category').cat.codes,
                    markers=mks, s=marker_size,
                    palette=sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex(), ax=axes[0])
    sns.scatterplot(x='state', y='meds_jitt', data=policy_df[policy_df['profile']==policy_df['profile'].unique()[1]],
                    hue=policy_df.trt.astype('category').cat.codes,
                    hue_order=np.arange(policy_df.trt.unique().shape[0]), style=policy_df.policy.astype('category').cat.codes,
                    markers=mks, s=marker_size,
                    palette=sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex(), legend=False, ax=axes[1])
    sns.scatterplot(x='state', y='meds_jitt', data=policy_df[policy_df['profile']==policy_df['profile'].unique()[2]],
                    hue=policy_df.trt.astype('category').cat.codes,
                    hue_order=np.arange(policy_df.trt.unique().shape[0]), style=policy_df.policy.astype('category').cat.codes,
                    markers=mks, s=marker_size,
                    palette=sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex(), legend=False, ax=axes[2])
    sns.scatterplot(x='state', y='meds_jitt', data=policy_df[policy_df['profile']==policy_df['profile'].unique()[3]],
                    hue=policy_df.trt.astype('category').cat.codes,
                    hue_order=np.arange(policy_df.trt.unique().shape[0]), style=policy_df.policy.astype('category').cat.codes,
                    markers=mks, s=marker_size,
                    palette=sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex(), legend=False, ax=axes[3])

    # Adding vertical lines to identify state classes in plots
    axes[0].vlines(x=[0.5, 2.5, 4.5], ymin=ylims[0], ymax=ylims[1], color='darkgray', linestyle='dotted', linewidth=0.5)
    axes[1].vlines(x=[0.5, 2.5, 4.5], ymin=ylims[0], ymax=ylims[1], color='darkgray', linestyle='dotted', linewidth=0.5)
    axes[2].vlines(x=[0.5, 2.5, 4.5], ymin=ylims[0], ymax=ylims[1], color='darkgray', linestyle='dotted', linewidth=0.5)
    axes[3].vlines(x=[0.5, 2.5, 4.5], ymin=ylims[0], ymax=ylims[1], color='darkgray', linestyle='dotted', linewidth=0.5)

    # Adding horizontal lines to identify action classes in plots
    axes[0].hlines(y=[0.5], xmin=xlims[0], xmax=xlims[1], color='darkgray', linestyle='dotted', linewidth=0.5) # y=[0.5, 1.5, 2.5, 3.5, 4.5]
    axes[1].hlines(y=[0.5], xmin=xlims[0], xmax=xlims[1], color='darkgray', linestyle='dotted', linewidth=0.5)
    axes[2].hlines(y=[0.5], xmin=xlims[0], xmax=xlims[1], color='darkgray', linestyle='dotted', linewidth=0.5)
    axes[3].hlines(y=[0.5], xmin=xlims[0], xmax=xlims[1], color='darkgray', linestyle='dotted', linewidth=0.5)

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('\nHealth Condition', fontsize=axes_size, fontweight='semibold')

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='', ylim=ylims, yticks=range(2), ylabel='') # yticks=range(6)
    fig.subplots_adjust(bottom=0.28)
    for k, ax in enumerate(fig.axes):
        plt.sca(ax)
        plt.xticks(rotation=45, fontsize=tick_size-1.5)
        plt.yticks(fontsize=tick_size)

        if k==0:
            plt.ylabel('Number of Medications', fontsize=axes_size, fontweight='semibold')

    ## Adding subtitles
    axes[0].set_title('Normal BP', fontsize=subtitle_size, fontweight='semibold')
    axes[1].set_title('Elevated BP', fontsize=subtitle_size, fontweight='semibold')
    axes[2].set_title('Stage 1 Hypertension', fontsize=subtitle_size, fontweight='semibold')
    axes[3].set_title('Stage 2 Hypertension', fontsize=subtitle_size, fontweight='semibold')

    # Modifying Legends
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    ## Marker legend
    handles = [f(m, 'black') for m in mks]
    labels = ['Class-Ordered Monotone', 'Monotone Optimal', 'Optimal']
    lgnd = axes[0].legend(loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(2.4, -0.65),
                          handles=handles, labels=labels, prop={'size': legend_size})
    # lgnd.legendHandles[0]._legmarker.set_markersize(3.5)
    # lgnd.legendHandles[1]._legmarker.set_markersize(3.5)
    # lgnd.legendHandles[2]._legmarker.set_markersize(3.5)

    # Adding data labels
    colors = sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex()
    labels = policy_df.trt.unique()[policy_df.trt.astype('category').cat.codes.unique().argsort()]
    for p in range(len(fig.axes[:-1])):
        tmp_df = policy_df[policy_df['profile']==policy_df['profile'].unique()[p]].reset_index()
        uneq_trt = np.where(tmp_df.trt[tmp_df.policy=='CMP'].reset_index(drop=True, inplace=False)!=tmp_df.trt[tmp_df.policy=='MP'].reset_index(drop=True, inplace=False), 1, 0)
        for row in range(tmp_df.shape[0]):
            if row in tmp_df.trt[tmp_df.policy=='CMP'].index:
                if uneq_trt[row-tmp_df.trt[tmp_df.policy!='CMP'].shape[0]] == 1:
                    t_loc = -0.18
            else:
                t_loc = 0.15
            axes[p].text(tmp_df.state_id[row], tmp_df.meds_jitt[row]+t_loc, tmp_df.trt[row],
                         horizontalalignment='center', fontsize=data_label_size, color=colors[int(np.where(tmp_df.trt[row]==labels)[0])], zorder=0)

    # Removing incorrectly placed text (run only when necessary
    tmp_df = policy_df[policy_df['profile'] == policy_df['profile'].unique()[3]].reset_index()
    ind = np.where((tmp_df.trt=='ACE') & (tmp_df.policy=='OP'))[0]
    for t in [axes[3].texts[x] for x in ind]:
        t.set_visible(False)

    # Saving plot
    fig.set_size_inches(6.5, 2)
    plt.savefig('policy_plot_year10.eps', bbox_inches='tight')
    plt.close()

# Function to plot the policies of selected patients over states for a single year
def plot_policies_state_rev(policy_df):

    # Figure parameters
    xlims = [-0.5, 5.5] # limit of the x-axis
    ylims = [-0.1, 1.35] # limits of the y-axis # [-0.5, 5.5]
    axes_size = 7 # font size of axes labels
    subtitle_size = 7.5 # font size for subplot titles
    tick_size = 6.5 # font size for tick labels
    legend_size = 6.5 # font size for legend labels
    data_label_size = 5 # font size of data labels
    marker_size = 10 # marker size

    # Making figure
    fig, axes = plt.subplots(nrows=2, ncols=3)
    axs = axes.ravel() # list of axes in a single dimension
    mks = ['^', 'o', 's']
    for h, i in enumerate(policy_df['profile'].unique()[2:]): # including only profiles with stage 1 and stage 2 hypertension
        for j, k in enumerate(['OP', 'CMP', 'MP']):
            # lg = np.where(h==0 & j==0, True, False).all()
            sns.scatterplot(x='state', y='meds_jitt', data=policy_df[(policy_df['profile']==i) & (policy_df['policy']==k)],
                            # style=policy_df.policy.astype('category').cat.codes, marker=mks[j],
                            s=marker_size,
                            hue=policy_df.trt.astype('category').cat.codes,
                            hue_order=np.arange(policy_df.trt.unique().shape[0]),
                            palette=sns.color_palette('viridis', policy_df.trt.unique().shape[0]).as_hex(), legend=False, ax=axes[h, j])

    # Adding horizontal and vertical lines to identify classes in plots
    for ax in axs:
        # Vertical line to identify state classes in plots
        ax.vlines(x=[0.5, 2.5, 4.5], ymin=ylims[0], ymax=ylims[1], color='darkgray', linestyle='dotted', linewidth=0.5)

        # Horizontal lines to identify action classes in plots
        ax.hlines(y=[0.5], xmin=xlims[0], xmax=xlims[1], color='darkgray', linestyle='dotted', linewidth=0.5) # y=[0.5, 1.5, 2.5, 3.5, 4.5]

    # Figure Configuration
    ## Configuration for the panel plot
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel('\n\n\nHealth Condition', fontsize=axes_size-0.5, fontweight='semibold')
    plt.ylabel('Number of Medications', fontsize=axes_size-0.5, fontweight='semibold')
    # plt.title('Stage 1 Hypertension\n\n', fontsize=subtitle_size+0.5, fontweight='semibold') # title for top plots
    # axs[4].set_title('Stage 2 Hypertension', fontsize=subtitle_size+0.5, fontweight='semibold') # title for bottom plots

    ## Axes configuration for every subplot
    plt.setp(axes, xlim=xlims, xlabel='', ylim=ylims, yticks=range(2), ylabel='') # yticks=range(6)
    fig.subplots_adjust(bottom=0.28)
    for k, ax in enumerate(axs):
        plt.sca(ax)
        plt.xticks(rotation=30, fontsize=tick_size-1.5)
        plt.yticks(fontsize=tick_size)

        # Full captions
        if k==1:
            plt.xlabel('(a) Stage 1 Hypertension', fontsize=axes_size-0.5) # , rotation=270, fontweight='semibold'
        elif k==4:
            plt.title(' ', fontsize=data_label_size)
            plt.xlabel('(b) Stage 2 Hypertension', fontsize=axes_size-0.5) # , rotation=270, fontweight='semibold'

        # # Subplot label
        # if k==1:
        #     plt.xlabel('(A)', fontsize=axes_size-0.5) # , rotation=270, fontweight='semibold'
        # elif k==4:
        #     plt.title(' ', fontsize=data_label_size)
        #     plt.xlabel('(B)', fontsize=axes_size-0.5) # , rotation=270, fontweight='semibold'

    ## Adding subtitles
    for j, k in enumerate(['Optimal', 'Class-Ordered Monotone', 'Optimal Monotone']):
        axs[j].set_title(k, fontsize=subtitle_size-0.5, fontweight='semibold')

    # Adding data labels
    colors = sns.color_palette("viridis", policy_df.trt.unique().shape[0]).as_hex()
    labels = policy_df.trt.unique()[policy_df.trt.astype('category').cat.codes.unique().argsort()]
    for r, i in enumerate(policy_df['profile'].unique()[2:]):
        tmp_df = policy_df[policy_df['profile']==i].reset_index() # including only profiles with stage 1 and stage 2 hypertension
        for c, p in enumerate(['OP', 'CMP', 'MP']):
            tmp_df1 = tmp_df[tmp_df['policy']==p].reset_index()
            for row in range(tmp_df1.shape[0]):
                axes[r, c].text(tmp_df1.state_id[row], tmp_df1.meds_jitt[row] + 0.1, tmp_df1.trt[row],
                                horizontalalignment='center', fontsize=data_label_size, color=colors[int(np.where(tmp_df1.trt[row]==labels)[0])], zorder=0)

    # Saving plot
    fig.set_size_inches(6.5, 3.2)
    fig.tight_layout(pad=0.05, w_pad=1) #
    plt.savefig('policy_plot_year10_rev-captionsMI.eps', bbox_inches='tight')
    plt.close()
