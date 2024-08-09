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
        plt.savefig("events_prevented.pdf", bbox_inches='tight')
    else:
        plt.savefig("qalys_saved.pdf", bbox_inches='tight') #, dpi=300

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
        plt.savefig("toptal_pi.pdf", bbox_inches='tight')
    else:
        plt.savefig("pi_per_capita.pdf", bbox_inches='tight') #, dpi=300

# Function to plot the policies of selected patients over states for a single year ############# update with new policies
def plot_policies_state(policy_df):

    # Figure parameters
    xlims = [-1.2, 6.2] # limit of the x-axis
    ylims = [-0.2, 4.5] # limits of the y-axis # [-0.5, 5.5]
    yticks = range(5) # sequence of ticks
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
        for j, k in enumerate(['OP', 'MP', 'MQL']):
            # lg = np.where(h==0 & j==0, True, False).all()
            sns.scatterplot(x='state', y='meds', data=policy_df[(policy_df['profile']==i) & (policy_df['policy']==k)],
                            s=marker_size, legend=False, ax=axes[h, j],
                            hue=policy_df.action.astype('category').cat.codes,
                            hue_order=np.arange(policy_df.action.unique().shape[0]),
                            palette=sns.color_palette('viridis', policy_df.action.unique().shape[0]).as_hex())

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
    plt.setp(axes, xlim=xlims, xlabel='', ylim=ylims, yticks=yticks, ylabel='') # yticks=range(6)
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

    # Adding data labels
    colors = sns.color_palette("viridis", policy_df.action.unique().shape[0]).as_hex()
    labels = policy_df.labels.unique()[policy_df.labels.astype('category').cat.codes.unique().argsort()]
    labels[np.where(labels == '0 SD/0 HD')[0][0]] = 'NT' # changing '0 SD/0 HD' to 'NT
    policy_df.loc[np.where(policy_df.labels == '0 SD/0 HD')[0], 'labels'] = 'NT'
    for r, i in enumerate(policy_df['profile'].unique()[2:]):
        tmp_df = policy_df[policy_df['profile'] == i].reset_index() # including only profiles with stage 1 and stage 2 hypertension
        for c, p in enumerate(['OP', 'MP', 'MQL']):
            tmp_df1 = tmp_df[tmp_df['policy'] == p].reset_index()
            for row in range(tmp_df1.shape[0]):
                axes[r, c].text(tmp_df1.state_id[row] + 0.5, tmp_df1.meds[row] + 0.25, tmp_df1.labels[row],
                                horizontalalignment='center', fontsize=data_label_size, color=colors[np.where(tmp_df1.labels[row]==labels)[0][0]],
                                zorder=0, rotation=40)

    ## Adding subtitles
    for j, k in enumerate(['Optimal', 'Optimal Monotone', 'Monotone Q-learning']):
        axs[j].set_title(k, fontsize=subtitle_size-0.5, fontweight='semibold')

    # Saving plot
    fig.set_size_inches(6.5, 3.2)
    fig.tight_layout(pad=0.05, w_pad=1)
    plt.savefig('policy_plot_year10.pdf', bbox_inches='tight')
    plt.close()
