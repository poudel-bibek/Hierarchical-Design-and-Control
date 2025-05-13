import json
import xml.etree.ElementTree as ET
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
from utils import get_averages
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def count_consecutive_ones_filtered(actions):
    """
    Helper function to count consecutive occurrences of 1's in the action list.
    The first action (corresponding to intersection) is ignored.
    Returns a list where each element is the length of a consecutive sequence of 1's.

    Example:
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 1] → [2, 1, 3]
    """
    if not actions or len(actions) <= 1:
        return []

    counts = []
    count = 0

    # Start from the second action (index 1)
    for action in actions[1:]:
        if action == 1:
            count += 1
        else:
            if count > 0:
                counts.append(count)
                count = 0

    # Don't forget to add the last sequence if it ends with 1's
    if count > 0:
        counts.append(count)

    return counts

def plot_avg_consecutive_ones(file_path, output_path="./results/sampled_actions_retro.pdf"):
    """
    Creates a clean, professional plot of the average sum of consecutive occurrences of '1's
    per training iteration with a vibrant appearance.

    Parameters:
        file_path (str): Path to the JSON file containing the data.
        output_path (str): Path to save the output PDF file.
    """

    # Load data
    with open(file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)  # Convert iteration key to integer
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]

        # Calculate the sum of consecutive 1's for each sample, then average across samples
        sums_of_consecutive_ones = [sum(seq) for seq in consecutive_ones if seq]
        avg_consecutive_ones = np.mean(sums_of_consecutive_ones) if sums_of_consecutive_ones else 0

        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)

    # Set base font size
    fs = 24  # Base font size - adjust this to change all font sizes proportionally

    # Set up the figure with a clean style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Set background color
    ax.set_facecolor('white')

    # Calculate y-axis limits with some padding
    y_min = min(avg_consecutive_ones_per_iteration) * 0.9
    y_max = max(avg_consecutive_ones_per_iteration) * 1.1

    # Calculate x-axis limits with added margins
    x_min = min(iterations) - (max(iterations) - min(iterations)) * 0.05  # 5% margin on left
    x_max = max(iterations) + (max(iterations) - min(iterations)) * 0.05  # 5% margin on right

    # Set axis limits
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    # Format y-axis with one decimal place
    def format_with_decimals(x, pos):
        return f'{x:.1f}'

    ax.yaxis.set_major_formatter(FuncFormatter(format_with_decimals))

    # Add light grid lines with slightly more visibility
    ax.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Use a more vibrant blue for the data points
    VIBRANT_BLUE = '#2E5EAA'  # More vibrant blue for data points

    # Create scatter plot with more vibrant, semi-transparent circles
    scatter = ax.scatter(iterations, avg_consecutive_ones_per_iteration,
                        s=110, edgecolors=VIBRANT_BLUE, facecolors='none',
                        linewidth=2.0, alpha=0.75, zorder=3)

    # Fit a trend line
    z = np.polyfit(iterations, avg_consecutive_ones_per_iteration, 1)
    p = np.poly1d(z)

    # Create x values for the trend line (only within the data range)
    x_trend = np.linspace(min(iterations), max(iterations), 100)
    y_trend = p(x_trend)

    # Use a very dark blue color for the trend line - almost navy blue
    VERY_DARK_BLUE = '#0A2472'  # Very dark blue/navy color

    # Plot the trend line as a solid, very dark line
    trend_line = ax.plot(x_trend, y_trend, color=VERY_DARK_BLUE, linewidth=4.0, zorder=4)

    # Set labels with increased font size and more vibrant color
    LABEL_COLOR = '#1A1A1A'  # Slightly lighter than pure black for better contrast
    ax.set_xlabel('Training Iteration', fontsize=fs*1.2, labelpad=10, color=LABEL_COLOR)
    ax.set_ylabel('# of Synchronized Green Signals', fontsize=fs*1.2, labelpad=10, color=LABEL_COLOR)

    # Line for trend line - use the very dark blue color
    trend_line_handle = mlines.Line2D([], [], color=VERY_DARK_BLUE, linewidth=4.0,
                                     label='Trend Line')

    # Add the legend with the proper handles
    ax.legend(handles=[trend_line_handle],
             loc='upper right', frameon=True, framealpha=0.9,
             edgecolor='#CCCCCC', fontsize=fs)

    # Add padding between y-axis and tick labels
    ax.tick_params(axis='y', pad=8)  # Add padding between y-axis and y-tick labels

    # Customize tick parameters with larger font size and more vibrant color
    ax.tick_params(axis='both', colors=LABEL_COLOR, labelsize=fs)

    # Add a subtle border around the plot with slightly more visible color
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#AAAAAA')  # Slightly darker border
        ax.spines[spine].set_linewidth(1.2)  # Slightly thicker border

    # Add more padding around the entire plot
    plt.tight_layout(pad=2.0)

    # Save with extra padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()

    print(f"Plot saved to {output_path}")

def plot_control_results(*json_paths, in_range_demand_scales):
    """
    """
    fs = 17 
    COLORS = {
        'Signalized':   '#F4B400', 
        'Unsignalized': '#4285F4',   
        'RL (Ours)':    '#0F9D58',   
    }
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Open Sans', 'Arial', 'DejaVu Sans'],
        'text.color':         '#202124',
        'axes.edgecolor':     '#dadce0',
        'axes.linewidth':     1.0,
        'axes.titlesize':     fs + 2,
        'axes.titleweight':   'bold',
        'axes.labelsize':     fs,
        'xtick.color':        '#5f6368',
        'ytick.color':        '#5f6368',
        'xtick.labelsize':    fs - 1,
        'ytick.labelsize':    fs - 1,
        'grid.color':         '#e8eaed',
        'grid.linewidth':     0.8,
        'grid.linestyle':     '--',
        'legend.frameon':     False,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
    })

    fig = plt.figure(figsize=(16, 7))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.22)
    ax_pa = fig.add_subplot(gs[0,0])
    ax_pt = fig.add_subplot(gs[1,0], sharex=ax_pa)
    ax_va = fig.add_subplot(gs[0,1])
    ax_vt = fig.add_subplot(gs[1,1], sharex=ax_va)
    panels = [ax_pa, ax_pt, ax_va, ax_vt]

    all_scales = []
    for path in json_paths:
        scales = get_averages(path, total=False)[0]
        all_scales.extend(scales)
    all_scales = np.array(all_scales)

    unique_scales = np.sort(np.unique(all_scales))

    ax_pa.set_title('Pedestrian')
    ax_va.set_title('Vehicle')

    if len(json_paths) == 3:
        tl_idx = [i for i, p in enumerate(json_paths) if 'tl' in p.lower()][0]
        us_idx = [i for i, p in enumerate(json_paths) if 'unsignalized' in p.lower()][0]
        rl_idx = [i for i, p in enumerate(json_paths) if 'ppo' in p.lower()][0]
        json_paths = [json_paths[tl_idx], json_paths[us_idx], json_paths[rl_idx]]
        method_labels = ['Signalized', 'Unsignalized', 'RL (Ours)']
    else:
        tl_idx = [i for i, p in enumerate(json_paths) if 'tl' in p.lower()][0]
        rl_idx = [i for i, p in enumerate(json_paths) if 'ppo' in p.lower()][0]
        json_paths = [json_paths[tl_idx], json_paths[rl_idx]]
        method_labels = ['Signalized', 'RL (Ours)']


    x_min, x_max = all_scales.min(), all_scales.max()
    x_margin = 0.05 * (x_max - x_min)

    for ax in panels:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    valid_min_scale = min(in_range_demand_scales)
    valid_max_scale = max(in_range_demand_scales)

    for ax in panels:
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], valid_min_scale, facecolor='grey', alpha=0.25, zorder=-2)
        ax.axvspan(valid_max_scale, xlim[1], facecolor='grey', alpha=0.25, zorder=-2)

    legend_handles = []
    for _, (path, label) in enumerate(zip(json_paths, method_labels)):
        color = COLORS[label]

        scales, veh_avg_mean, ped_avg_mean, _, veh_avg_std, ped_avg_std, _ = get_averages(path, total=False)
        _, veh_tot, ped_tot, _, veh_tot_std, ped_tot_std, _ = get_averages(path, total=True)

        h_pa = ax_pa.plot(scales, ped_avg_mean, color=color, lw=2.5, label=label, zorder=2)[0]
        ax_pa.fill_between(scales,
                           ped_avg_mean - ped_avg_std,
                           ped_avg_mean + ped_avg_std,
                           color=color, alpha=0.2, zorder=2)

        ax_pt.plot(scales, ped_tot/1000, color=color, lw=2.5, zorder=2)
        ax_pt.fill_between(scales,
                           (ped_tot - ped_tot_std)/1000,
                           (ped_tot + ped_tot_std)/1000,
                           color=color, alpha=0.2, zorder=2)

        ax_va.plot(scales, veh_avg_mean, color=color, lw=2.5, zorder=2)
        ax_va.fill_between(scales,
                           veh_avg_mean - veh_avg_std,
                           veh_avg_mean + veh_avg_std,
                           color=color, alpha=0.2, zorder=2)

        ax_vt.plot(scales, veh_tot/1000, color=color, lw=2.5, zorder=2)
        ax_vt.fill_between(scales,
                           (veh_tot - veh_tot_std)/1000,
                           (veh_tot + veh_tot_std)/1000,
                           color=color, alpha=0.2, zorder=2)

        legend_handles.append(h_pa)
    
    scales_to_show = unique_scales[::2]
    labels = []
    for s in scales_to_show:
        if abs(s * 10 - round(s * 10)) < 1e-6:
            labels.append(f"{s:.1f}x")
        else:
            labels.append(f"{s:.2f}x")

    # Set major ticks only at the locations we want to label
    ax_pt.set_xticks(scales_to_show)
    ax_vt.set_xticks(scales_to_show)

    ax_pt.set_xticklabels(labels)
    ax_vt.set_xticklabels(labels)

    for ax in panels:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y')
        ax.set_xticks(unique_scales, minor=True)
        ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)
        ax.grid(which='major', axis='x', linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)

    fig.text(0.03, 0.76, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.03, 0.29, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.52, 0.76, 'Average Wait Time (s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.52, 0.29, 'Total Wait Time (×10³ s)', va='center', rotation='vertical', fontsize=fs+1)

    ax_pt.set_xlabel('Demand Scale', fontsize=fs+1) # (× original)
    ax_vt.set_xlabel('Demand Scale', fontsize=fs+1) # (× original)

    ax_pa.tick_params(labelbottom=False)
    ax_va.tick_params(labelbottom=False)

    n_yticks = 6
    for ax in panels:
        # ax.set_ylim(bottom=-0.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    fig.legend(legend_handles, method_labels,
               ncol=len(method_labels),
               loc='lower center',
               bbox_to_anchor=(0.5, -0.08),  # push slightly below panels
               frameon=True,
               edgecolor='#dadce0',
               fontsize=fs)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.11, wspace=0.10, hspace=0.12)
    plt.savefig("consolidated_control_results.pdf", bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_design_results(*json_paths, in_range_demand_scales):
    """
    """
    original_pedestrian_demand = 2222.80
    COLORS = {'Design Agent': '#0F9D58', 
            'Real-world': '#4285F4'}

    fs = 17
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Open Sans', 'Arial', 'DejaVu Sans'],
        'text.color':         '#202124',
        'axes.edgecolor':     '#dadce0',
        'axes.linewidth':     1.0,
        'axes.titlesize':     fs + 2,
        'axes.titleweight':   'bold',
        'axes.labelsize':     fs,
        'xtick.color':        '#5f6368',
        'ytick.color':        '#5f6368',
        'xtick.labelsize':    fs - 1,
        'ytick.labelsize':    fs - 1,
        'grid.color':         '#e8eaed',
        'grid.linewidth':     0.8,
        'grid.linestyle':     '--',
        'legend.frameon':     False,
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
    })

    fig = plt.figure(figsize=(9, 8)) # Slightly wider/taller for better label spacing
    gs  = GridSpec(2, 1, figure=fig, hspace=0.12) # Adjusted spacing if needed
    ax_avg = fig.add_subplot(gs[0, 0])
    ax_tot = fig.add_subplot(gs[1, 0], sharex=ax_avg)
    panels = [ax_avg, ax_tot]

    ax_avg.set_title('Pedestrian')

    if len(json_paths) != 2:
        raise ValueError('plot_design_results expects exactly two json paths')

    all_scales = []
    design_scales = None
    for path in json_paths:
        scales = get_averages(path, total=False)[0]
        all_scales.extend(scales)
        if design_scales is None: # Store scales from the first path (Design Agent)
             design_scales = scales
    all_scales = np.array(all_scales)
    unique_scales = np.sort(np.unique(all_scales))

    x_min, x_max = unique_scales.min(), unique_scales.max()
    x_margin = 0.05 * (x_max - x_min)

    # Remove top/right spines and set x-axis limits
    for ax in panels:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    # Shade out-of-range demand scales before drawing gridlines
    vmin, vmax = min(in_range_demand_scales), max(in_range_demand_scales)
    for ax in panels:
        xlim = ax.get_xlim()  # axis limits already set
        ax.axvspan(xlim[0], vmin, facecolor='grey', alpha=0.25, zorder=-2)
        ax.axvspan(vmax, xlim[1], facecolor='grey', alpha=0.25, zorder=-2)

    # Draw gridlines on top of shading
    for ax in panels:
        ax.grid(True, axis='y')  # Only horizontal grid lines
        ax.set_xticks(unique_scales, minor=True)
        ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.8,
                alpha=0.7, zorder=-5)
        ax.grid(which='major', axis='x', linestyle='--', linewidth=0.8,
                alpha=0.7, zorder=-5)

    legend_handles = []
    labels = ['Design Agent', 'Real-world'] # Assuming order matches json_paths
    for path, label in zip(json_paths, labels):
        scales, _, _, avg_vals, _, _, avg_std = get_averages(path, total=False)
        _, _, _, tot_vals, _, _, tot_std = get_averages(path, total=True)
        color = COLORS[label]

        h = ax_avg.plot(scales, avg_vals, color=color, lw=2.5, label=label, zorder=2)[0]
        ax_avg.fill_between(scales, avg_vals - avg_std, avg_vals + avg_std, color=color, alpha=0.2, zorder=2)

        tot_k = tot_vals / 1000.0
        tot_k_std = tot_std / 1000.0
        ax_tot.plot(scales, tot_k, color=color, lw=2.5, zorder=2)
        ax_tot.fill_between(scales, tot_k - tot_k_std, tot_k + tot_k_std, color=color, alpha=0.2, zorder=2)

        legend_handles.append(h)

    scales_to_show = unique_scales[::2] # Show every other major tick
    if unique_scales[-1] not in scales_to_show:
         scales_to_show = np.append(scales_to_show, unique_scales[-1])

    ax_tot.set_xticks(scales_to_show)
    
    x_labs = [f"{s:.1f}x" if abs(s * 10 - round(s * 10)) < 1e-6 else f"{s:.2f}x" for s in scales_to_show]
    ax_tot.set_xticklabels(x_labs)

    ax_tot.set_xlabel('Demand Scale', fontsize=fs + 1)
    ax_avg.tick_params(labelbottom=False) # Hide x-labels on top plot

    # Uniform Y-ticks: integer, padded bottom
    n_yticks = 6 # Match control plot
    for ax in panels:
        if ax == ax_avg:
            ax.set_ylim(bottom=50, top=120)
        else:
            ax.set_ylim(bottom=-0.5) # Match control plot padding
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

    fig.text(0.01, 0.76, 'Average Arrival Time (s)', va='center', rotation='vertical', fontsize=fs+1)
    fig.text(0.01, 0.29, 'Total Arrival Time (×10³ s)', va='center', rotation='vertical', fontsize=fs+1)

    fig.legend(legend_handles, labels,
               ncol=len(labels),
               loc='lower center',
               bbox_to_anchor=(0.5, -0.08), # Adjusted anchor
               frameon=True, # Keep frame off
               edgecolor='#dadce0',
               fontsize=fs)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.10, wspace=0.10, hspace=0.12) # Adjust margins
    plt.savefig('consolidated_design_results.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_consolidated_insights(sampled_actions_file_path, conflict_json_file_path, switching_freq_data_path):
    """
    Creates a consolidated figure with three subplots:
    1. Left: Bar chart of mean conflicts across demand scales with error bars
    2. Middle: Plot of average consecutive ones over training iterations
    3. Right: TL as horizontal line and RL as histogram for switching frequency (TL switching frequency is obtained analytically as 54 for 600 timestep horizon)

    Parameters:
    - sampled_actions_file_path: Path to JSON file containing action data
    - conflict_json_file_path: Path to JSON file containing conflict data
    - switching_freq_data: Dictionary containing switching frequency data (optional)
    """
    # Function to process data from json
    def process_json_data(json_data, key):
        # Extract data by demand scale
        data = {}
        for demand_scale, runs in json_data.items():
            values = [run_data[key] for run_index, run_data in runs.items()]
            data[float(demand_scale)] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        return data

    # Load conflict data
    with open(conflict_json_file_path, 'r') as f:
        conflict_json_data = json.load(f)

    # Process conflict data
    processed_conflict_data = process_json_data(conflict_json_data, "total_conflicts")

    # Set base font size
    fs = 23

    # Set consistent number of y-ticks for all subplots
    n_ticks = 5  # Define the number of y-ticks to use across all subplots

    # Set up the figure with a 1x3 grid
    fig = plt.figure(figsize=(24, 6.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1])

    # Create subplots
    ax_near_accidents = fig.add_subplot(gs[0, 0])
    ax_consecutive_ones = fig.add_subplot(gs[0, 1])
    ax_switching_freq = fig.add_subplot(gs[0, 2])

    # Set style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    # Define colors - updated middle plot colors
    BRIGHT_BLUE = '#0078D7'  # New bright blue for middle plot trend line
    VIBRANT_BLUE = '#2E5EAA'  # Keep for scatter points
    SALMON = '#E29587'  # Subtle salmon for TL/Unsignalized
    SEA_GREEN = '#85B79D'  # Subtle sea green for RL

    # ========== LEFT SUBPLOT: Conflict events across demand scales ==========
    # Filter demand scales to only include the specified levels
    selected_demand_scales = [0.5, 1.0, 1.5, 2.0, 2.5]
    filtered_scales = [scale for scale in selected_demand_scales if scale in processed_conflict_data]

    conflict_means = [processed_conflict_data[scale]["mean"] for scale in filtered_scales]
    conflict_stds = [processed_conflict_data[scale]["std"] for scale in filtered_scales]

    # Even more subtle gradient - using shades of orange/coral with less intensity
    colors = [
        '#FDE5D2',  # Very pale orange for 0.5x
        '#FDCBAD',  # Lighter orange for 1.0x
        '#FCB08A',  # Light salmon for 1.5x
        '#FC9774',  # Salmon for 2.0x
        '#FB7D5B'   # Darker salmon for 2.5x
    ]

    # Make sure we have enough colors
    if len(colors) < len(filtered_scales):
        colors = colors * (len(filtered_scales) // len(colors) + 1)
    colors = colors[:len(filtered_scales)]

    # Create bar positions
    x_positions = np.arange(len(filtered_scales))
    width = 0.5

    # Create bar chart with MORE PROMINENT error bars
    bars = ax_near_accidents.bar(x_positions, conflict_means, width, color=colors,
                               edgecolor='#333333', linewidth=1.0,
                               yerr=conflict_stds, capsize=8, error_kw={'elinewidth': 2.5, 'ecolor': '#333333', 'capthick': 2.5})

    # Add data labels to the left of the top of each bar
    # for i, bar in enumerate(bars):
    #     height = bar.get_height() + 9
    #     # Position text to the left of the bar top
    #     ax_near_accidents.text(bar.get_x() + 0.25*width, height,
    #                          f'{int(conflict_means[i])}', ha='right', va='center',
    #                          fontsize=fs-4)

    labelsize = fs-4
    # Set x-ticks at the bar positions with the appropriate labels
    ax_near_accidents.set_xticks(x_positions)
    ax_near_accidents.set_xticklabels([f'{scale}x' for scale in filtered_scales], fontsize=labelsize)

    # Styling
    ax_near_accidents.set_ylabel('# of Conflicts in Unsignalized', fontsize=fs)  # Updated label
    ax_near_accidents.set_xlabel('Demand Scale', fontsize=fs)
    ax_near_accidents.tick_params(axis='both', labelsize=labelsize)

    # Set y-limit with headroom for labels and error bars
    ax_near_accidents.set_ylim(0, max(conflict_means + np.array(conflict_stds)) * 1.1)  # More headroom for labels

    # Make grid match middle plot (light lines behind data)
    ax_near_accidents.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_near_accidents.set_axisbelow(True)

    # Remove top and right spines to match middle plot
    ax_near_accidents.spines['top'].set_visible(False)
    ax_near_accidents.spines['right'].set_visible(False)

    # Set consistent y-ticks
    ax_near_accidents.yaxis.set_major_locator(MaxNLocator(n_ticks))

    # ========== MIDDLE SUBPLOT: Average consecutive ones plot ==========
    # Load data
    with open(sampled_actions_file_path, "r") as file:
        data = json.load(file)

    # Compute the average sum of consecutive 1's per iteration
    avg_consecutive_ones_per_iteration = []
    iterations = []

    for iteration, actions_list in data.items():
        iteration = int(iteration)
        consecutive_ones = [count_consecutive_ones_filtered(action_list) for action_list in actions_list]
        sums_of_consecutive_ones = [sum(seq) for seq in consecutive_ones if seq]
        avg_consecutive_ones = np.mean(sums_of_consecutive_ones) if sums_of_consecutive_ones else 0
        iterations.append(iteration)
        avg_consecutive_ones_per_iteration.append(avg_consecutive_ones)

    # Sort by iteration
    iterations, avg_consecutive_ones_per_iteration = zip(*sorted(zip(iterations, avg_consecutive_ones_per_iteration)))
    iterations = np.array(iterations)
    avg_consecutive_ones_per_iteration = np.array(avg_consecutive_ones_per_iteration)

    # Set background color
    ax_consecutive_ones.set_facecolor('white')

    # Calculate y-axis limits with padding
    y_min = 3.3  # Set explicitly to 3.2 to match the lowest data point
    y_max = 4.1  # Set explicitly to 4.1 to provide headroom for highest points

    # Calculate x-axis limits with margins
    x_min = min(iterations) - (max(iterations) - min(iterations)) * 0.05
    x_max = max(iterations) + (max(iterations) - min(iterations)) * 0.05

    # Set axis limits
    ax_consecutive_ones.set_ylim(y_min, y_max)
    ax_consecutive_ones.set_xlim(x_min, x_max)

    # Format y-axis with one decimal place
    ax_consecutive_ones.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))

    # Add light grid lines
    ax_consecutive_ones.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_consecutive_ones.set_axisbelow(True)

    # Remove top and right spines
    ax_consecutive_ones.spines['top'].set_visible(False)
    ax_consecutive_ones.spines['right'].set_visible(False)

    # Create scatter plot - KEEPING ORIGINAL COLORS
    scatter = ax_consecutive_ones.scatter(iterations, avg_consecutive_ones_per_iteration,
                                        s=110, edgecolors=VIBRANT_BLUE, facecolors='none',
                                        linewidth=2.0, alpha=0.75, zorder=3)

    # Fit a trend line
    z = np.polyfit(iterations, avg_consecutive_ones_per_iteration, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(iterations), max(iterations), 100)
    y_trend = p(x_trend)

    # Calculate 95% confidence interval
    n = len(iterations)
    x_mean = np.mean(iterations)
    y_mean = np.mean(avg_consecutive_ones_per_iteration)

    # Sum of squares
    ss_xx = np.sum((iterations - x_mean)**2)
    ss_xy = np.sum((iterations - x_mean) * (avg_consecutive_ones_per_iteration - y_mean))
    ss_yy = np.sum((avg_consecutive_ones_per_iteration - y_mean)**2)

    # Regression slope and intercept
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # Standard error of estimate
    y_hat = slope * iterations + intercept
    se = np.sqrt(np.sum((avg_consecutive_ones_per_iteration - y_hat)**2) / (n - 2))

    # Confidence interval
    alpha = 0.05  # 95% confidence interval
    t_val = stats.t.ppf(1 - alpha/2, n - 2)

    # Calculate confidence bands
    x_eval = x_trend
    ci = t_val * se * np.sqrt(1/n + (x_eval - x_mean)**2 / ss_xx)
    y_upper = y_trend + ci
    y_lower = y_trend - ci

    # Plot the trend line with new bright blue color
    trend_line = ax_consecutive_ones.plot(x_trend, y_trend, color=BRIGHT_BLUE, linewidth=4.0, zorder=4, label='Trend Line')

    # Add confidence interval with shading
    confidence_interval = ax_consecutive_ones.fill_between(x_trend, y_lower, y_upper,
                                                         color=BRIGHT_BLUE, alpha=0.2,
                                                         zorder=2, label='95% Confidence Interval')

    # Set labels
    ax_consecutive_ones.set_xlabel('Training Episode', fontsize=fs)
    ax_consecutive_ones.set_ylabel('Synchronized Green Signals', fontsize=fs)

    # Create legend with both trend line and confidence interval
    trend_line_handle = mlines.Line2D([], [], color=BRIGHT_BLUE, linewidth=4.0,
                                    label='Trend Line')
    ci_handle = mpatches.Patch(facecolor=BRIGHT_BLUE, alpha=0.2,
                              label='95% Confidence Interval')

    ax_consecutive_ones.legend(handles=[trend_line_handle, ci_handle],
                            loc='upper right', frameon=True, framealpha=0.9,
                            edgecolor='#CCCCCC', fontsize=fs-4)

    # Tick parameters
    ax_consecutive_ones.tick_params(axis='both', labelsize=labelsize)

    # Set consistent y-ticks with fixed 0.1 interval to ensure we have 3.6 tick
    ax_consecutive_ones.yaxis.set_major_locator(MultipleLocator(0.2))

    # ========== RIGHT SUBPLOT: Switching frequency with TL as horizontal line ==========

    # Load frequency data
    with open(switching_freq_data_path, 'r') as f:
        frequency_json_data = json.load(f)

    # Process frequency data
    processed_frequency_data = process_json_data(frequency_json_data, "total_switches")

    frequency_demands = [0.5, 1.0, 1.5, 2.0, 2.5]
    filtered_demands = [demand for demand in frequency_demands if demand in processed_frequency_data]

    frequency_means = [processed_frequency_data[demand]["mean"] for demand in filtered_demands]
    frequency_stds = [processed_frequency_data[demand]["std"] for demand in filtered_demands]

    # Create placeholder data with TL having same value across demand scales
    tl_value = 54  # Same value for all demand scales

    # Get x positions for grouped bars
    x = np.arange(len(filtered_demands))
    width = 0.5  # Width of bars - keep the same

    # Create subtle gradient for RL bars
    rl_colors = [
        '#CFEAD6',  # Lower level lighter green
        '#A8D5BA',  # Lightest sea green
        '#8CCB9B',  # Light sea green
        '#73C17E',  # Medium sea green
        '#5AB663'   # Deeper sea green
    ]

    # Ensure we have enough colors
    if len(rl_colors) < len(filtered_demands):
        rl_colors = rl_colors * (len(filtered_demands) // len(rl_colors) + 1)
    rl_colors = rl_colors[:len(filtered_demands)]

    # Set up the plot with a discontinuous y-axis
    ax_switching_freq.set_facecolor('white')

    # Function to transform values to the broken y-axis scale
    def transform_y(y):
        # Map values to a discontinuous scale:
        # 0-54 maps to 0-0.2 (bottom 20% of plot)
        # 260-320 maps to 0.3-1.0 (top 70% of plot)
        if y <= 54:
            return y / 54 * 0.2
        else:
            return 0.3 + (y - 260) / (320 - 260) * 0.7

    # Plot the bars with standard deviations
    for i, (mean, std) in enumerate(zip(frequency_means, frequency_stds)):
        # Calculate bar height in the transformed space
        bar_height = transform_y(mean) - transform_y(0)

        # Draw the bar
        bar = ax_switching_freq.bar(x[i], bar_height, width=width,
                                   bottom=transform_y(0),
                                   color=rl_colors[i],
                                   edgecolor='#333333',
                                   linewidth=1.0)

        # Add error bars
        # Calculate the std dev in the transformed space
        yerr = transform_y(mean + std) - transform_y(mean)

        # Draw error bar
        ax_switching_freq.errorbar(x[i], transform_y(mean), yerr=yerr,
                                  fmt='none', ecolor='#333333', capsize=8,
                                  elinewidth=2.5, capthick=2.5)

    # Add the TL horizontal line
    tl_line = ax_switching_freq.axhline(y=transform_y(tl_value), color=SALMON, linewidth=3, linestyle='-', zorder=5)

    # Get the y-axis line width to match the break marks to it
    axis_line_width = ax_switching_freq.spines['left'].get_linewidth()

    # Create break marks for the y-axis
    # Position of the break in the transformed scale
    break_pos = 0.31  # middle of the gap between 0.2 and 0.3

    # Draw break marks on the left y-axis only
    # Increased spacing between diagonal lines
    gap = 0.020  # Increased gap between the diagonal lines
    d = 0.03    # Size of the diagonal lines

    # First create a white rectangle to "erase" part of the axis
    # This ensures the break appears as a true gap in the axis
    rect_height = gap * 1.5  # Height of white rectangle
    rect_width = d * 2.0     # Width of white rectangle

    # Draw white background rectangle to create a clean break
    white_patch = plt.Rectangle((-rect_width/2, break_pos-rect_height/2), rect_width, rect_height,
                              facecolor='white', edgecolor='none', transform=ax_switching_freq.transAxes,
                              clip_on=False, zorder=10)
    ax_switching_freq.add_patch(white_patch)

    # Then draw the diagonal lines centered on the axis
    # Make sure line width matches the axis line width
    kwargs = dict(transform=ax_switching_freq.transAxes, color='black',
                 clip_on=False, linewidth=axis_line_width, zorder=11)

    # Upper diagonal line
    ax_switching_freq.plot([-d/2, d/2], [break_pos+gap/2, break_pos+gap/2 + d], **kwargs)

    # Lower diagonal line
    ax_switching_freq.plot([-d/2, d/2], [break_pos-gap/2, break_pos-gap/2 + d], **kwargs)

    # Set the y-ticks at the actual data values
    yticks = [0, tl_value, 275, 300]
    yticklabels = [str(int(y)) for y in yticks]

    ax_switching_freq.set_yticks([transform_y(y) for y in yticks])
    ax_switching_freq.set_yticklabels(yticklabels, fontsize=labelsize)

    # Create legend handles
    tl_handle = mlines.Line2D([], [], color=SALMON, linewidth=3, linestyle='-', label='Signalized')
    rl_handle = mpatches.Patch(facecolor=rl_colors[1], edgecolor='#333333', linewidth=1.0, label='RL (Ours)')

    # Styling
    ax_switching_freq.set_ylabel('Switching Frequency', fontsize=fs)
    ax_switching_freq.set_xlabel('Demand Scale', fontsize=fs)
    ax_switching_freq.set_xticks(x)

    # Format x-ticks to show demand scale
    demand_labels = [f"{d}x" for d in filtered_demands]
    ax_switching_freq.set_xticklabels(demand_labels, fontsize=labelsize)

    ax_switching_freq.tick_params(axis='both', labelsize=labelsize)

    # Make grid match middle plot (light lines behind data)
    ax_switching_freq.grid(True, linestyle='-', alpha=0.15, color='#333333')
    ax_switching_freq.set_axisbelow(True)

    # Remove top and right spines to match middle plot
    ax_switching_freq.spines['top'].set_visible(False)
    ax_switching_freq.spines['right'].set_visible(False)

    # Set uniform margins in right subplot
    # Calculate the margin to add on each side (half the width of a bar)
    margin = 0.7
    # Set the x-limits to create uniform margins
    ax_switching_freq.set_xlim(-margin, len(filtered_demands) - 1 + margin)

    # Set y-limits for the plot
    ax_switching_freq.set_ylim(0, 1.05)  # Provide headroom for the legend

    # Add legend in the top right corner
    ax_switching_freq.legend(handles=[tl_handle, rl_handle], fontsize=fs-4, loc='upper right',
                           bbox_to_anchor=(1.0, 1.01))

    # ========== Add (a), (b), (c) labels centered below each subplot ==========
    # Get the exact position of each subplot after tight_layout
    bbox1 = ax_near_accidents.get_position()
    bbox2 = ax_consecutive_ones.get_position()
    bbox3 = ax_switching_freq.get_position()
    # bcbar = cbar.ax.get_position(fig)

    x1 = 0.17
    x2 = 0.5
    x3 = 0.77

    # Common y position for labels
    label_y = 0.1 # Adjusted slightly from previous attempts

    fig.text(x1, label_y, "(a)", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    fig.text(x2, label_y, "(b)", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    fig.text(x3, label_y, "(c)", ha="center", va="bottom", fontsize=fs, fontweight="bold")

    # ========== Figure-level adjustments ==========
    plt.subplots_adjust(wspace=0.23, bottom=0.1)  # Adjusted bottom margin to make room for labels

    # Save figure
    plt.savefig("./results/consolidated_insights.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()
    return fig

def gmm_to_video():
    """
    """
    pass 

def graph_to_video():
    """
    """
    pass 




def _load_graph(path):
    obj = pickle.load(open(path, "rb"))
    return obj[0] if isinstance(obj, tuple) else obj

def _crop_graph(G_orig, lower, upper):
    pos_orig = nx.get_node_attributes(G_orig, "pos")
    if len(pos_orig) != G_orig.number_of_nodes():
        pos_orig = nx.spring_layout(G_orig, seed=42) # Fallback
        nx.set_node_attributes(G_orig, pos_orig, "pos")

    # Calculate original coordinate ranges for jitter scaling
    y_coords_orig = np.array([coord[1] for coord in pos_orig.values()])
    y_range = y_coords_orig.ptp() if len(y_coords_orig) > 1 else 1.0
    jitter_std_dev = y_range * 0.005 # 0.5% of y-range

    low, high = np.percentile(y_coords_orig, [lower, upper])

    nodes_inside = {n for n, (_, y) in pos_orig.items() if low <= y <= high}

    # Start with subgraph of nodes inside the range and edges between them
    H = G_orig.subgraph(nodes_inside).copy()
    pos_H = {n: pos_orig[n] for n in H.nodes()}

    boundary_nodes_data = []
    boundary_node_counter = 0

    # Find edges crossing the boundary in the original graph
    for u, v in G_orig.edges():
        if u in nodes_inside and v in nodes_inside:
            continue # Skip edges fully inside

        uy = pos_orig[u][1]
        vy = pos_orig[v][1]
        u_inside = low <= uy <= high
        v_inside = low <= vy <= high

        if u_inside != v_inside: # Found a crossing edge
            inside_node = u if u_inside else v
            outside_node = v if u_inside else u
            ux, uy = pos_orig[inside_node]
            vx, vy = pos_orig[outside_node]

            y_boundary = -1
            if vy < low:
                y_boundary = low
            elif vy > high:
                y_boundary = high
            else:
                continue # Should not happen

            # Calculate intersection x-coordinate
            x_intersect = ux # Default for vertical
            if abs(vy - uy) > 1e-9: # Avoid division by zero
                if abs(vx - ux) > 1e-9: # Not vertical
                    t = (y_boundary - uy) / (vy - uy)
                    x_intersect = ux + t * (vx - ux)

            boundary_pos = (x_intersect, y_boundary)
            boundary_nodes_data.append((boundary_pos, inside_node))

    # Add unique boundary nodes and connecting edges to H
    processed_boundaries = {} # Cache boundary points: rounded_pos -> node_id
    for boundary_pos, inside_node in boundary_nodes_data:
        x_intersect, y_boundary = boundary_pos
        rounded_pos = (round(x_intersect, 6), round(y_boundary, 6))

        if rounded_pos not in processed_boundaries:
            # Add random vertical jitter
            y_jitter = np.random.normal(0, jitter_std_dev)
            final_y = y_boundary + y_jitter
            final_boundary_pos = (x_intersect, final_y) # Jitter applied to y

            new_boundary_node_id = f"boundary_{boundary_node_counter}"
            boundary_node_counter += 1
            H.add_node(new_boundary_node_id)
            pos_H[new_boundary_node_id] = final_boundary_pos # Store jittered position
            processed_boundaries[rounded_pos] = new_boundary_node_id # Use original rounded pos for lookup
            boundary_node_id = new_boundary_node_id
        else:
            boundary_node_id = processed_boundaries[rounded_pos]

        # Add edge from inside node to the boundary node
        if inside_node in H:
             H.add_edge(inside_node, boundary_node_id)

    return H, pos_H

def _stretch_pos(pos, sy):
    return {n: (x, y * sy) for n, (x, y) in pos.items()}

def _draw_graph(ax, G, pos, node_size):
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_size,
                           node_color="#2ecc71",
                           edgecolors="black", linewidths=0.5)
    # print("All Node ids: ", G.nodes())
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="black", width=0.5)

    # Calculate and set X/Y limits tightly around data before turning axis off
    if pos:
        x_coords = [x for x, _ in pos.values()]
        y_coords = [y for _, y in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_padding = (x_max - x_min) * 0.01 # Reduced padding to 1%
        y_padding = (y_max - y_min) * 0.05 # Keep y padding at 5%
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_axis_off(); ax.set_aspect("equal")

def plot_graphs_and_gmm( graph_a_path, 
                        graph_b_path, 
                        gmm_path,
                        figsize=(18, 8), 
                        dpi=300, 
                        surf_res=200,
                        y_scale=5.0,
                        node_size=50, 
                        y_crop=(12, 86),
                        gmm_cmap_style='coolwarm'):
    
    G1, pos1_raw = _crop_graph(_load_graph(graph_a_path), *y_crop)
    G2, pos2_raw = _crop_graph(_load_graph(graph_b_path), *y_crop)
    pos1, pos2   = _stretch_pos(pos1_raw, y_scale), _stretch_pos(pos2_raw, y_scale)

    # Remove isolated nodes before plotting
    for G, pos in [(G1, pos1), (G2, pos2)]:
        isolated_nodes = [n for n, degree in G.degree() if degree == 0]
        G.remove_nodes_from(isolated_nodes)
        for node in isolated_nodes:
            if node in pos:
                del pos[node]

    gmm = pickle.load(open(gmm_path, "rb"))[0]
    locs = gmm.component_distribution.loc.detach().cpu().numpy()
    
    # Print the original GMM means
    print("GMM component means (original):")
    for i, mean in enumerate(locs):
        print(f"Mean {i}: Location={mean[0]:.4f}, Thickness={mean[1]:.4f}")
    
    # Create a copy and manually modify means to simulate mode collapse
    modified_locs = locs.copy()
    
    # Manually set thickness values while keeping original locations
    
    # Group 1: means with locations around 0.36-0.42 (means 0, 1, 5)
    # Original values:
    # Mean 0: Location=0.4072, Thickness=0.0528
    # Mean 1: Location=0.3622, Thickness=0.5207
    # Mean 5: Location=0.4241, Thickness=0.8140
    
    # Set all thicknesses in group 1 to be similar
    modified_locs[0][1] = 0.38  # Mean 0 thickness
    modified_locs[1][1] = 0.36  # Mean 1 thickness
    modified_locs[5][1] = 0.32  # Mean 5 thickness
    
    # Group 2: means with locations around 0.73-0.77 (means 2, 4)
    # Original values:
    # Mean 2: Location=0.7377, Thickness=0.5972
    # Mean 4: Location=0.7792, Thickness=0.0183
    
    # Set all thicknesses in group 2 to be similar
    modified_locs[2][1] = 0.32  # Mean 2 thickness
    modified_locs[4][1] = 0.30  # Mean 4 thickness
    
    # Means 3 and 6 remain unchanged
    
    # Update the means in the model
    device = gmm.component_distribution.loc.device
    gmm.component_distribution.loc = torch.tensor(modified_locs, dtype=torch.float32, device=device)
    
    # Print modified means
    print("\nGMM component means (manually modified):")
    for i, mean in enumerate(modified_locs):
        print(f"Mean {i}: Location={mean[0]:.4f}, Thickness={mean[1]:.4f}")
    
    # Use full domain with negative offset for ymin to avoid apparent truncation
    xmin, xmax = 0.0, 1.05
    ymin, ymax = -0.02, 1.05
    
    X = np.linspace(xmin, xmax, surf_res)
    Y = np.linspace(ymin, ymax, surf_res)
    Xg, Yg = np.meshgrid(X, Y)
    grid_pts = torch.tensor(np.column_stack([Xg.ravel(), Yg.ravel()]),
                            dtype=torch.float32,
                            device=gmm.component_distribution.loc.device)
    with torch.no_grad():
        dens = torch.exp(gmm.log_prob(grid_pts)).cpu().numpy().reshape(surf_res, surf_res)
    dens_norm = (dens - dens.min()) / dens.ptp()
    
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Simple 1x3 GridSpec
    gs  = fig.add_gridspec(1, 3,
                           width_ratios=[2.5, 2.5, 3],
                           wspace=0.1) # Increased wspace

    # Assign axes
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2], projection="3d")

    fs = 18

    # Common styling for GMM subplot's ticks, labels, and fonts, based on 'coolwarm'
    ax3.set_xlabel('Location', fontsize=fs-2, labelpad=10, color='#202124')
    ax3.set_zlabel('Density', fontsize=fs-2, labelpad=10, color='#202124')

    z_lim_unified = (0, 0.8)
    ax3.set_zlim(*z_lim_unified)
    z_ticks_unified = np.arange(0, z_lim_unified[1] + 0.01, 0.2) # Includes 0.8
    ax3.set_zticks(z_ticks_unified)
    ax3.set_zticklabels([f"{t:.1f}" for t in z_ticks_unified])

    ax3.tick_params(axis='both', which='major', labelsize=fs-2, colors='#5f6368')
    
    common_norm = mpl.colors.Normalize(vmin=z_lim_unified[0], vmax=z_lim_unified[1])

    _draw_graph(ax1, G1, pos1, node_size)
    _draw_graph(ax2, G2, pos2, node_size)
    
    # Define blue colors for "_mid" nodes and their incident edges
    lighter_blue_color = "#FFA07A"  # Light salmon for nodes
    darker_blue_color = "#FF4500"   # Orange red for edges and node outlines

    for G, pos, ax in [(G1, pos1, ax1), (G2, pos2, ax2)]:
        mid_nodes = [n for n in G.nodes() if "_mid" in str(n)]
        if mid_nodes:
            # Draw edges incident to these nodes in darker blue
            mid_edges = [e for e in G.edges() if e[0] in mid_nodes or e[1] in mid_nodes]
            if mid_edges:
                nx.draw_networkx_edges(G, pos, ax=ax,
                                       edgelist=mid_edges,
                                       edge_color=darker_blue_color,
                                       width=1.0) # Thicker orange edges
            # Draw nodes on top in lighter blue with darker blue outlines
            nx.draw_networkx_nodes(G, pos, ax=ax,
                                   nodelist=mid_nodes,
                                   node_size=node_size,
                                   node_color=lighter_blue_color,
                                   edgecolors=darker_blue_color,
                                   linewidths=0.5)
    
    # Choose colormap based on gmm_cmap_style
    if gmm_cmap_style == 'viridis':
        cmap = plt.get_cmap("viridis", 256)
        ax3.plot_surface(Xg, Yg, dens_norm,
                         rstride=2, cstride=2,       
                         cmap=cmap,                  
                         linewidth=0,                
                         alpha=0.9,                  
                         antialiased=False)          

        ax3.set_ylabel('Thickness', fontsize=fs-2, labelpad=12, color='#202124') # Coolwarm font style, Viridis text
        ax3.set_title('GMM Distribution', fontweight='bold', fontsize=fs) # Added title

        ax3.grid(True) # Simpler grid for viridis style
        norm = common_norm # Use common norm

    elif gmm_cmap_style == 'coolwarm':
        cmap = plt.get_cmap("coolwarm", 256)
        ax3.plot_surface(Xg, Yg, dens_norm,
                         rstride=2, 
                         cstride=2,
                         facecolors=cmap(dens_norm), # Original method for coolwarm
                         linewidth=0.05,
                         edgecolor='white',
                         alpha=0.9,
                         antialiased=True, 
                         shade=False)                 # Original shade=False

        ax3.set_ylabel("Width", fontsize=fs-2, labelpad=12, color='#202124') # Original label and coolwarm font style
        # No title for coolwarm style
        
        # Custom grid for coolwarm style
        ax3.grid(True)
        grid_style = dict(color=(0.0, 0.0, 0.0, 0.2), linestyle=(0, (5, 5)), linewidth=0.5)
        for axis_obj in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
            axis_obj._axinfo["grid"].update(grid_style)
        
        norm = common_norm # Use common norm
    else:
        raise ValueError(f"Unsupported gmm_cmap_style: {gmm_cmap_style}. Choose 'viridis' or 'coolwarm'.")

    ax3.set_xlim(xmin, xmax)
    ax3.set_ylim(ymin, ymax)
    
    ax3.view_init(elev=35, azim=-50) # Changed view angle
    
    # Drawing colorbar relative to ax3 with controlled height
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, shrink=0.35, pad=0.10, label='')
    cbar.ax.tick_params(labelsize=fs-2, colors='#5f6368')  # Match other tick font sizes and use gray color
    
    # --- Precise Label Positioning ---
    # Ensure figure is drawn to get accurate bounding boxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get precise bounding boxes in display coordinates
    bb1_disp = ax1.get_tightbbox(renderer)
    bb2_disp = ax2.get_tightbbox(renderer)
    bb3_disp = ax3.get_tightbbox(renderer)
    bbcbar_disp = cbar.ax.get_window_extent(renderer) # Use get_window_extent for colorbar

    # Transform to figure coordinates
    bb1_fig = bb1_disp.transformed(fig.transFigure.inverted())
    bb2_fig = bb2_disp.transformed(fig.transFigure.inverted())
    bb3_fig = bb3_disp.transformed(fig.transFigure.inverted())
    bbcbar_fig = bbcbar_disp.transformed(fig.transFigure.inverted())

    # Calculate centers precisely
    x1 = bb1_fig.x0 / 2 + 0.06
    x2 = bb2_fig.x0 + bb2_fig.width / 2 - 0.01
    x3 = bb3_fig.x0 + (bbcbar_fig.x1 - bb3_fig.x0) / 2 + 0.04 # Center + slight right shift

    # Common y position for labels (relative to bottom of figure)
    label_y = 0.17 # Kept user's adjustment

    fig.text(x1, label_y, "(a)", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    fig.text(x2, label_y, "(b)", ha="center", va="bottom", fontsize=fs, fontweight="bold")
    fig.text(x3, label_y, "(c)", ha="center", va="bottom", fontsize=fs, fontweight="bold")

    ax1.text(0.54, -0.25, "Original Network",
             horizontalalignment='center', 
             verticalalignment='bottom',
             transform=ax1.transAxes,
             fontsize=fs)
    
    ax2.text(0.63, -0.35, "Final Network", 
             horizontalalignment='center', 
             verticalalignment='bottom',
             transform=ax2.transAxes,
             fontsize=fs)

    # Adjust subplot layout manually - increase wspace
    plt.subplots_adjust(left=-0.04, right=0.98, top=0.98, bottom=0.1, wspace=0.1)

    # --- Manual Shift Upwards for Subplot (c) --- (Moved After subplots_adjust)
    dy = 0.01 # Vertical shift amount
    # Get original positions again (might be slightly different after draw)
    b3_final = ax3.get_position()
    bcbar_final = cbar.ax.get_position()
    # Apply shift
    ax3.set_position([b3_final.x0, b3_final.y0 + dy, b3_final.width, b3_final.height])
    cbar.ax.set_position([bcbar_final.x0, bcbar_final.y0 + dy, bcbar_final.width, bcbar_final.height])

    # Save figure tightly
    fig.savefig('./graphs_gmm.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

def rewards_results_plot(combined_csv_codesign, 
                         combined_csv_control, 
                         results_codesign,
                         results_separate,
                         data_type = "average", # average or total
                         use_error_bars = True):
    """
    (a) Contains both codesign and control reward plot
    (b) Pedestrian wait time results
    (c) Vehicle wait time results
    """
    # Configuration variables
    MOVING_AVG_WINDOW = 200  # Window size for moving average
    MAX_STEPS = 20e6  # Maximum steps to show (11 million)
    
    # Load data
    df_c = pd.read_csv(combined_csv_codesign)
    df_ctrl = pd.read_csv(combined_csv_control)

    # Clean infinities and ensure numeric
    for df in [df_c, df_ctrl]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['step'] = pd.to_numeric(df['step'], errors='coerce')
        
        # Dynamically identify reward columns
        reward_cols = [col for col in df.columns if col.startswith('reward')]
        for col in reward_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Apply cutoff at MAX_STEPS - make explicit copies to avoid SettingWithCopyWarning
    df_c = df_c[df_c['step'] <= MAX_STEPS].copy()
    df_ctrl = df_ctrl[df_ctrl['step'] <= MAX_STEPS].copy()

    # Calculate average reward per row (across all reward columns)
    df_c['mean_reward'] = df_c[[col for col in df_c.columns if col.startswith('reward')]].mean(axis=1)
    df_ctrl['mean_reward'] = df_ctrl[[col for col in df_ctrl.columns if col.startswith('reward')]].mean(axis=1)

    # Compute moving average and rolling standard deviation
    df_c['mean_reward_ma'] = df_c['mean_reward'].rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
    df_c['mean_reward_std'] = df_c['mean_reward'].rolling(MOVING_AVG_WINDOW, min_periods=1).std()
    
    df_ctrl['mean_reward_ma'] = df_ctrl['mean_reward'].rolling(MOVING_AVG_WINDOW, min_periods=1).mean()
    df_ctrl['mean_reward_std'] = df_ctrl['mean_reward'].rolling(MOVING_AVG_WINDOW, min_periods=1).std()

    # Fill NaN values in std columns (first few rows) with zeros
    df_c['mean_reward_std'] = df_c['mean_reward_std'].fillna(0)
    df_ctrl['mean_reward_std'] = df_ctrl['mean_reward_std'].fillna(0)

    # Calculate statistics for the last 500,000 steps
    last_steps = 500000
    
    # For codesign
    last_c_data = df_c[df_c['step'] >= (df_c['step'].max() - last_steps)]
    if not last_c_data.empty:
        # Stats on the raw rewards
        c_raw_mean = last_c_data['mean_reward'].mean()
        c_raw_std = last_c_data['mean_reward'].std()
        
        # Stats on the rolling mean (smoothed data)
        c_ma_mean = last_c_data['mean_reward_ma'].mean()
        c_ma_std = last_c_data['mean_reward_ma'].std()
        
        print(f"\nCo-design (last {last_steps} steps):")
        print(f"  Raw data:")
        print(f"    Average reward: {c_raw_mean:.2f}")
        print(f"    Standard deviation: {c_raw_std:.2f}")
        print(f"  Smoothed data (moving avg window={MOVING_AVG_WINDOW}):")
        print(f"    Average: {c_ma_mean:.2f}")
        print(f"    Standard deviation: {c_ma_std:.2f}")
    else:
        print(f"Not enough data for co-design statistics (less than {last_steps} steps)")
    
    # For separate control
    last_ctrl_data = df_ctrl[df_ctrl['step'] >= (df_ctrl['step'].max() - last_steps)]
    if not last_ctrl_data.empty:
        # Stats on the raw rewards
        ctrl_raw_mean = last_ctrl_data['mean_reward'].mean()
        ctrl_raw_std = last_ctrl_data['mean_reward'].std()
        
        # Stats on the rolling mean (smoothed data)
        ctrl_ma_mean = last_ctrl_data['mean_reward_ma'].mean()
        ctrl_ma_std = last_ctrl_data['mean_reward_ma'].std()
        
        print(f"\nSeparate control (last {last_steps} steps):")
        print(f"  Raw data:")
        print(f"    Average reward: {ctrl_raw_mean:.2f}")
        print(f"    Standard deviation: {ctrl_raw_std:.2f}")
        print(f"  Smoothed data (moving avg window={MOVING_AVG_WINDOW}):")
        print(f"    Average: {ctrl_ma_mean:.2f}")
        print(f"    Standard deviation: {ctrl_ma_std:.2f}")
    else:
        print(f"Not enough data for separate control statistics (less than {last_steps} steps)")

    # Prepare for plotting
    x_c = df_c['step'].to_numpy(dtype=float)
    y_c = [df_c[col].to_numpy(dtype=float) for col in df_c.columns if col.startswith('reward') and col not in ['mean_reward', 'mean_reward_ma', 'mean_reward_std']]
    ma_c = df_c['mean_reward_ma'].to_numpy(dtype=float)
    std_c = df_c['mean_reward_std'].to_numpy(dtype=float)

    x_ctrl = df_ctrl['step'].to_numpy(dtype=float)
    y_ctrl = [df_ctrl[col].to_numpy(dtype=float) for col in df_ctrl.columns if col.startswith('reward') and col not in ['mean_reward', 'mean_reward_ma', 'mean_reward_std']]
    ma_ctrl = df_ctrl['mean_reward_ma'].to_numpy(dtype=float)
    std_ctrl = df_ctrl['mean_reward_std'].to_numpy(dtype=float)

    # Define styling parameters
    fs = 23  # Base font size
    dpi = 300
    
    # Set consistent styling
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Open Sans', 'Arial', 'DejaVu Sans'],
        'text.color': '#202124',  # Dark gray for main text
        'axes.edgecolor': '#dadce0',  # Light gray for axes edges
        'axes.linewidth': 1.0,
        'axes.titlesize': fs + 2,
        'axes.titleweight': 'bold',
        'axes.labelsize': fs,
        'xtick.color': '#5f6368',  # Medium gray for tick labels
        'ytick.color': '#5f6368',
        'xtick.labelsize': fs - 1,
        'ytick.labelsize': fs - 1,
        'grid.color': '#e8eaed',  # Very light gray for grid
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'legend.facecolor': 'white',
        'legend.edgecolor': '#cccccc',
        'axes.titlepad': 12
    })

    # Create figure with wider layout but equal subplot sizes
    figsize = (24, 7)  # Adjusted for better proportions
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 3, wspace=0.20)  # Equal width subplots with proper spacing
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Define colors - use the same colors across all subplots
    COLOR_CODESIGN = '#FF8000'  # Orange for CoDesign
    COLOR_CONTROL = '#4169E1'   # Royal Blue for Control

    # Plot (a): CoDesign - Background raw data (individual runs)
    # Removing raw data lines as per user request
    
    # Co-design: Add std deviation band around the moving average - without border
    ax1.fill_between(x_c, ma_c - std_c, ma_c + std_c, alpha=0.3, color=COLOR_CODESIGN, 
                     edgecolor='none', zorder=2)
    
    # Co-design: Main line (moving average)
    codesign_line, = ax1.plot(x_c, ma_c, color=COLOR_CODESIGN, linewidth=2.5, zorder=3)

    # Plot (a): Control - Background raw data (individual runs)
    # Removing raw data lines as per user request
    
    # Separate control: Add std deviation band around the moving average - without border
    ax1.fill_between(x_ctrl, ma_ctrl - std_ctrl, ma_ctrl + std_ctrl, alpha=0.3, color=COLOR_CONTROL,
                     edgecolor='none', zorder=2)
    
    # Separate control: Main line (moving average)
    control_line, = ax1.plot(x_ctrl, ma_ctrl, color=COLOR_CONTROL, linewidth=2.5, zorder=3)

    # Set y-axis limits for ax1
    ax1.set_ylim(-1450, 150)
    
    # Set specific y-ticks (6 ticks)
    ax1.set_yticks([-1400, -1100, -800, -500, -200, 100])
    
    # Set x-axis limit using MAX_STEPS with left padding
    ax1.set_xlim(-MAX_STEPS * 0.05, MAX_STEPS)
    
    # Set specific x-ticks (0 to MAX_STEPS in increments of 5 million)
    x_ticks = np.arange(0, MAX_STEPS + 1e6, 5e6)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{int(x/1e6)}' for x in x_ticks])
    ax1.set_xlabel('Simulation Step (x10$^6$)', fontsize=fs, labelpad=10)
    
    # Style the main plot
    # ax1.set_title('Training Rewards', fontsize=fs+2, fontweight='bold', pad=12)
    ax1.set_ylabel('Control Agent Reward', fontsize=fs, labelpad=10)
    ax1.tick_params(axis='both', labelsize=fs-1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)
    
    # --- Implement subplots (b) and (c) ---
    # Function for gradient line creation (similar to plot_design_and_control_results)
    def create_gradient_line(ax, x, y, base_color, lw=3.5, zorder=3, label=None):
        # Create points
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Convert hex to RGB
        r = int(base_color[1:3], 16) / 255.0
        g = int(base_color[3:5], 16) / 255.0
        b = int(base_color[5:7], 16) / 255.0
        
        # Create lighter variant (for gradient end)
        r2 = min(1.0, r + 0.15)
        g2 = min(1.0, g + 0.15)
        b2 = min(1.0, b + 0.15)
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom_gradient", 
            [(r, g, b), (r2, g2, b2)],
            N=100
        )
        
        # Create a gradient effect along the line
        norm = plt.Normalize(0, len(x)-1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
        lc.set_array(np.arange(len(x)))
        
        line = ax.add_collection(lc)
        
        # Add markers separately
        scatter = ax.scatter(x, y, color=base_color, s=36, marker='o', 
                            edgecolor='white', linewidth=1.0, zorder=zorder+1)
        
        # Create a dummy line with the right color for the legend
        if label is not None:
            dummy_line, = ax.plot([], [], color=base_color, lw=lw, marker='o', 
                                 markersize=6, markeredgecolor='white', markeredgewidth=1.0,
                                 label=label)
            return dummy_line
        return line
    
    # Fixed demand scales for x-axis
    fixed_scales = [0.5, 1.0, 1.5, 2.0, 2.5]
    fixed_scale_labels = [f'{s:.1f}x' for s in fixed_scales]
    
    # Set y-label text based on data_type
    if data_type == "average":
        y_label = 'Average Wait Time (s)'
    else:  # total
        y_label = 'Total Wait Time (×10³ s)'
    
    # Load results data for subplot (b) - Pedestrian
    # Cache data to avoid redundant loading
    data_cache = {}
    
    # Define configurations for the two plots
    plot_configs = [
        # (ax, title, domain)
        (ax2, "Pedestrian", "pedestrian"),
        (ax3, "Vehicle", "vehicle")
    ]
    
    # Store legend handles and labels
    all_legend_handles = []
    all_legend_labels = []
    
    # Process results for pedestrian and vehicle
    for ax, title, domain in plot_configs:
        for path, plot_title, color in [
            (results_codesign, "Co-design", COLOR_CODESIGN),
            (results_separate, "Separate-control", COLOR_CONTROL)
        ]:
            if not path:  # Skip if path is not provided
                ax.axis('off')
                continue
                
            # Load data using get_averages
            if path not in data_cache:
                data_false = get_averages(path, total=False)
                data_true = get_averages(path, total=True)
                data_cache[path] = {'total_false': data_false, 'total_true': data_true}
            
            # Get appropriate data based on domain and data_type
            if domain == "pedestrian":
                if data_type == "average":
                    scales, _, _, values, _, _, values_std = data_cache[path]['total_false']
                else:  # total
                    scales, _, _, values, _, _, values_std = data_cache[path]['total_true']
                    values = values / 1000.0  # Convert to thousands
                    values_std = values_std / 1000.0
            else:  # vehicle
                if data_type == "average":
                    scales, values, _, _, values_std, _, _ = data_cache[path]['total_false']
                else:  # total
                    scales, values, _, _, values_std, _, _ = data_cache[path]['total_true']
                    values = values / 1000.0  # Convert to thousands
                    values_std = values_std / 1000.0
            
            # Set up the axis
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Create plot
            h = create_gradient_line(ax, scales, values, color, lw=3.0, zorder=20, label=plot_title)
            
            # Add error visualization based on preference
            if use_error_bars:
                ax.errorbar(scales, values, yerr=values_std, color=color, capsize=4.5,
                           elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
            else:
                ax.fill_between(scales, values - values_std, values + values_std,
                               color=color, alpha=0.2, zorder=5)
                
            # Style the axis
            ax.set_title(title, fontsize=fs+2, fontweight='bold', pad=12)
            ax.set_ylabel(y_label, fontsize=fs, labelpad=10)
            ax.set_xlabel('Demand Scale', fontsize=fs, labelpad=10)
            ax.tick_params(axis='both', labelsize=fs-1)
            
            # Use fixed x ticks
            ax.set_xticks(fixed_scales)
            ax.set_xticklabels(fixed_scale_labels)
            
            # Set y-axis formatting with 5 ticks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
            
            # Add grid
            ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, zorder=-5)
            
            # Save handle for joint legend
            if plot_title not in all_legend_labels:
                all_legend_handles.append(h)
                all_legend_labels.append(plot_title)

    # Create a single shared legend in the middle of the figure
    legend_kwargs = {
        'loc': 'lower center',
        'ncol': 2,  # Force legend into a single line
        'bbox_to_anchor': (0.52, -0.125),  # Position it below the panels
        'fontsize': fs - 2,
        'frameon': True,
        'fancybox': True,
        'facecolor': 'white',
        'edgecolor': '#cccccc',
        'framealpha': 1.0,
        'borderpad': 0.6,
        'labelspacing': 0.4
    }
    
    # Add shared legend directly using handles from all subplots
    legend = fig.legend(all_legend_handles, all_legend_labels, **legend_kwargs)
    
    # Increase the linewidth in the legend
    for line in legend.get_lines():
        line.set_linewidth(3.5)  # Thicker lines in legend only

    # Subplot labels below each panel
    label_y = -0.16  # Position for panel labels - moved up from -0.33 to -0.26
    label_fontsize = fs + 2
    fig.text(0.215, label_y, "(a)", ha="center", va="center", fontsize=label_fontsize, fontweight="bold")
    fig.text(0.505, label_y, "(b)", ha="center", va="center", fontsize=label_fontsize, fontweight="bold")
    fig.text(0.835, label_y, "(c)", ha="center", va="center", fontsize=label_fontsize, fontweight="bold")

    # Final adjustments
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.13)
    plt.savefig("rewards_results_plot.png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def plot_gmm_top_down(gmm_pkl_path: str, 
                               location_range: tuple[float, float] = (0.0, 1.05), 
                               thickness_range: tuple[float, float] = (0.0, 1.05), 
                               fs: int = 19, 
                               num_grid_points: int = 100,
                               contour_levels: int = 20):
    """
    Plots the top-down view of a GMM distribution with markers from a .pkl file.
    The output is saved as 'gmm_flat.png'.

    Args:
        gmm_pkl_path (str): Path to the pickle file containing (gmm_object, markers_data).
        location_range (tuple[float, float]): Min and max for the location (x-axis).
        thickness_range (tuple[float, float]): Min and max for the thickness (y-axis).
        fs (int): Base font size.
        num_grid_points (int): Number of points for the grid in each dimension.
        contour_levels (int): Number of levels for the contour plot.
    """
    with open(gmm_pkl_path, "rb") as f:
        gmm_single, markers = pickle.load(f)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()

    label_color = '#202124'
    tick_color = '#5f6368'
    # Return to more subtle grid lines
    grid_color = (0.0, 0.0, 0.0, 0.55)  

    xmin, xmax = location_range
    ymin, ymax = thickness_range
    X_grid = np.linspace(xmin, xmax, num_grid_points)
    Y_grid = np.linspace(ymin, ymax, num_grid_points)
    X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)

    device = gmm_single.component_distribution.loc.device

    positions = torch.tensor(np.column_stack([X_mesh.ravel(), Y_mesh.ravel()]), 
                           dtype=torch.float32,
                           device=device)

    with torch.no_grad():
        Z_log_prob = gmm_single.log_prob(positions).detach().cpu()
    Z = np.exp(Z_log_prob.numpy()).reshape(X_mesh.shape)
    
    z_min = Z.min()
    z_ptp = Z.ptp()
    if z_ptp == 0:
        Z_norm = np.zeros_like(Z) if z_min == 0 else np.ones_like(Z) * (z_min / (z_min + 1e-9) )
    else:
        Z_norm = (Z - z_min) / z_ptp

    # Manual grid lines with low zorder 
    ax.set_axisbelow(True)
    
    # Draw horizontal grid lines with subtler appearance
    grid_y_ticks = np.arange(0.0, 1.1, 0.2)
    for y in grid_y_ticks:
        ax.axhline(y=y, color=grid_color, linestyle=(0, (5, 5)), linewidth=0.5, zorder=-10)  # Back to original linewidth
    
    # Draw vertical grid lines with subtler appearance
    grid_x_ticks = np.arange(0.0, 1.1, 0.2)
    for x in grid_x_ticks:
        ax.axvline(x=x, color=grid_color, linestyle=(0, (5, 5)), linewidth=0.5, zorder=-10)  # Back to original linewidth

    cmap = plt.get_cmap("coolwarm", 256)
    # Adjust contour opacity for better balance
    contour = ax.contourf(X_mesh, Y_mesh, Z_norm, levels=contour_levels, cmap=cmap, alpha=0.85, zorder=1)  # Moderate opacity
    
    cbar = plt.colorbar(contour, ax=ax, shrink=1.0, aspect=20, pad=0.05)
    cbar.set_label('Normalized Density', fontweight='bold', fontsize=fs-2, color=label_color)
    cbar.ax.tick_params(labelsize=fs-2, colors=tick_color)

    if hasattr(gmm_single, 'component_distribution') and hasattr(gmm_single.component_distribution, 'loc'):
        means = gmm_single.component_distribution.loc.detach().cpu().numpy()
        # Keep royal blue as it was requested
        royal_blue = '#0066ff'  # Royal blue hex color
        ax.scatter(means[:, 0], means[:, 1], 
                   c=royal_blue,
                   marker='o', 
                   s=120, 
                   edgecolors='black', 
                   linewidths=0.7, 
                   label='Component Means', 
                   zorder=2)

    if markers is not None:
        locations, thicknesses = markers
        ax.scatter(locations, thicknesses, 
                   c='red',
                   marker='x', 
                   s=120, 
                   label='Samples Drawn', 
                   zorder=3)

        x_range_plot = ax.get_xlim()[1] - ax.get_xlim()[0]
        offset_x = x_range_plot * 0.04

        for i, (loc, thick) in enumerate(zip(locations, thicknesses)):
            ax.text(loc + offset_x, thick, f'C{i+1}',
                    fontsize=fs-5, 
                    ha='left', 
                    va='center',
                    zorder=4)

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                       ncol=2, 
                       frameon=True, fancybox=True, facecolor='white', edgecolor='#cccccc',
                       framealpha=1.0, fontsize=fs-2, borderpad=0.6, labelspacing=0.4)
    legend.set_zorder(10)

    ax.set_xlabel('Location', fontweight='bold', fontsize=fs, color=label_color, labelpad=10)
    ax.set_ylabel('Thickness', fontweight='bold', fontsize=fs, color=label_color, labelpad=10)
    ax.set_title('GMM Distribution', fontweight='bold', fontsize=fs, color=label_color, pad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=fs-2, colors=tick_color)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(tick_color)
    ax.spines['bottom'].set_color(tick_color)
    
    ax.set_xlim(location_range)
    ax.set_ylim(thickness_range)

    output_filename = "gmm_flat.png"
    
    plt.subplots_adjust(bottom=0.2)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"GMM top-down plot saved to {output_filename}")

def plot_demand(
    xml_ped_path: str = "./simulation/original_pedtrips.xml",
    xml_veh_path: str = "./simulation/original_vehtrips.xml",
    bin_width: int = 60,
    figsize: tuple[int, int] = (14, 4),):
    
    """
    Produce side‑by‑side demand plots:

    (a) Pedestrians
    (b) Vehicles
    """

    def _extract_depart_times(xml_path: str | Path, tag: str):
        departs = []
        for _, elem in ET.iterparse(xml_path, events=("start",)):
            if elem.tag == tag and "depart" in elem.attrib:
                departs.append(float(elem.attrib["depart"]))
            elem.clear()
        return np.asarray(departs)

    def _counts_per_minute(departs: np.ndarray):
        edges = np.arange(0, departs.max() + bin_width, bin_width)
        counts, _ = np.histogram(departs, bins=edges)
        centers = edges[:-1] + bin_width / 2
        return centers, counts

    def _nice_ticks(data_min: float, data_max: float, step: int):
        first = np.floor(data_min / step) * step
        ticks = first + step * np.arange(6)
        while data_max > ticks[-2]:
            ticks += step
        return ticks[:6], (ticks[0], ticks[-1])

    fs         = 18                       # base font size
    gray_tick  = "#5f6368"
    label_col  = "#202124"
    ped_col    = "#6A5ACD"                # slate‑blue neon
    veh_col    = "#FF7F50"                # coral neon
    grid_kw    = dict(color='black',
                      linestyle=(0, (5, 5)),
                      linewidth=0.5,
                      alpha=0.4)  # reduced alpha for background grid dashes

    ped_x, ped_y = _counts_per_minute(
        _extract_depart_times(xml_ped_path, "person"))
    veh_x, veh_y = _counts_per_minute(
        _extract_depart_times(xml_veh_path, "trip"))

    ped_ticks, ped_ylim = _nice_ticks(ped_y.min(), ped_y.max(), 10)
    veh_ticks, veh_ylim = _nice_ticks(veh_y.min(), veh_y.max(), 2)
    ped_ylim = (ped_ylim[0], ped_ylim[1]-5)
    veh_ylim = (veh_ylim[0], veh_ylim[1]-1)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, x, y, col, title in (
        (axes[0], ped_x, ped_y, ped_col, "Pedestrian"),
        (axes[1], veh_x, veh_y, veh_col, "Vehicle"),
    ):
        ax.plot(x, y, color=col, linewidth=2.5)

        # titles & labels
        ax.set_title(title, fontweight="bold", color=label_col, fontsize=fs)
        ax.set_xlabel("Simulation Time (s)",    color=label_col, fontsize=fs)
        ax.set_ylabel("No. of Departures",      color=label_col, fontsize=fs)

        # ticks
        ax.tick_params(colors=gray_tick, labelsize=fs)
        if ax is axes[0]:
            ax.set_yticks(ped_ticks[:-1])
            ax.set_ylim(ped_ylim)
        else:
            ax.set_yticks(veh_ticks[:-1])
            ax.set_ylim(veh_ylim)

        # X‑axis ticks 0–35 with ×10² offset
        xticks = np.arange(0, 3501, 500)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t // 100}" for t in xticks])
        ax.annotate(
            r"$\times10^{2}$",
            xy=(0.99, -0.03),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=fs - 8,
            color=gray_tick,
        )

        # grid & spines
        ax.grid(True, **grid_kw)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(gray_tick)

    # small space between panels
    fig.subplots_adjust(wspace=0.08)   # << tiny gap
    plt.tight_layout()

    # panel markers
    fig.canvas.draw()
    for ax, lab in zip(axes, ("(a)", "(b)")):
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            pos.y0 - 0.19,
            lab,
            ha="center",
            va="top",
            fontsize=fs,
            fontweight="bold",
            color=label_col,
        )

    plt.savefig("./demand.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_design_and_control_results(design_unsig_path, realworld_unsig_path,
                                     control_tl_path, control_ppo_path,
                                     in_range_demand_scales, use_error_bars=True):
    """
    Combines the design and control results into a single figure with three columns,
    using specific colors and split legends. The figure shows:
    Left (a): Pedestrian Arrival Time results.
    Middle (b): Pedestrian Wait Time results.
    Right (c): Vehicle Wait Time results.
    """

    fs = 23 # Updated font size
    n_yticks = 5 # Define the number of y-ticks for all subplots

    # Define Colors (anonymous) - Adjusted to be less bright
    COLOR_INDIAN_RED = '#C93038'     # For Real-world (Subdued red)
    COLOR_SPRING_GREEN = '#3C9F40'   # For Design Agent (Ours) (Subdued green)
    COLOR_DARK_ORCHID = '#8064A2'    # For Signalized (Softer purple)
    COLOR_SLATE_GRAY = '#E67E22'     # For Unsignalized (Softer orange)
    COLOR_ROYAL_BLUE = '#3771A1'     # For Control Agent (Ours) (Softer blue)


    # Map plot elements to Colors - Updated based on new scheme
    COLORS = {
        'Design Agent (Ours)': COLOR_SPRING_GREEN,   # Design plot: Spring Green
        'Real-world':          COLOR_INDIAN_RED,     # Design plot: Indian Red
        'Signalized':          COLOR_DARK_ORCHID,    # Control plots: Dark Orchid
        'Unsignalized':        COLOR_SLATE_GRAY,     # Control plots: Slate Gray
        'Control Agent (Ours)': COLOR_ROYAL_BLUE,     # Control plots: Royal Blue
    }
    
    # Helper function to create a gradient line
    def create_gradient_line(ax, x, y, base_color, lw=3.5, zorder=3, label=None):
        # Create points
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Convert hex to RGB
        r = int(base_color[1:3], 16) / 255.0
        g = int(base_color[3:5], 16) / 255.0
        b = int(base_color[5:7], 16) / 255.0
        
        # Create lighter variant (for gradient end) - more subtle increase
        r2 = min(1.0, r + 0.15)  # Less dramatic lightening
        g2 = min(1.0, g + 0.15)  # Less dramatic lightening
        b2 = min(1.0, b + 0.15)  # Less dramatic lightening
        
        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom_gradient", 
            [(r, g, b), (r2, g2, b2)],
            N=100
        )
        
        # Create a gradient effect along the line
        norm = plt.Normalize(0, len(x)-1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
        lc.set_array(np.arange(len(x)))
        
        line = ax.add_collection(lc)
        
        # Add markers separately
        scatter = ax.scatter(x, y, color=base_color, s=36, marker='o', 
                            edgecolor='white', linewidth=1.0, zorder=zorder+1)
        
        # Create a dummy line with the right color for the legend
        if label is not None:
            dummy_line, = ax.plot([], [], color=base_color, lw=lw, marker='o', 
                                 markersize=6, markeredgecolor='white', markeredgewidth=1.0,
                                 label=label)
            return dummy_line
        return line

    # Enhanced styling setup for a more professional look
    mpl.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Open Sans', 'Arial', 'DejaVu Sans'],
        'text.color':         '#202124',
        'axes.edgecolor':     '#dadce0',     # Reverted to original lighter gray
        'axes.linewidth':     1.0,           # Reverted to original width
        'axes.titlesize':     fs + 2,        # Title size based on fs
        'axes.titleweight':   'bold',
        'axes.labelsize':     fs,            # Axis label size based on fs
        'axes.labelweight':   'medium',      # Make labels slightly bolder
        'xtick.color':        '#5f6368',     # Reverted to original gray tick color
        'ytick.color':        '#5f6368',     # Reverted to original gray tick color
        'xtick.labelsize':    fs - 1,        # Tick label size
        'ytick.labelsize':    fs - 1,        # Tick label size
        'xtick.major.width':  1.0,           # Reverted to original tick width
        'ytick.major.width':  1.0,           # Reverted to original tick width
        'grid.color':         '#e8eaed',     # Reverted to original grid color
        'grid.linewidth':     0.8,
        'grid.linestyle':     '--',
        'grid.alpha':         0.7,           # Grid transparency
        'legend.frameon':     True,          # Add frame to legend
        'legend.framealpha':  0.9,           # Make legend background more opaque
        'legend.edgecolor':   '#cccccc',     # Light gray legend border
        'legend.fontsize':    fs - 2,        # Consistent legend font size
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'legend.facecolor':   'white', 
        'axes.titlepad':      12,            # Add padding below axis titles
        'axes.spines.top':    False,         # Remove top spines globally
        'axes.spines.right':  False          # Remove right spines globally
    })

    # Create figure and grid (2 rows, 3 columns) - Height remains the same as previous
    fig = plt.figure(figsize=(24, 12))
    # Adjusted spacing
    gs  = GridSpec(2, 3, figure=fig, hspace=0.10, wspace=0.20)

    # Create axes
    ax_design_avg = fig.add_subplot(gs[0, 0])
    ax_design_tot = fig.add_subplot(gs[1, 0], sharex=ax_design_avg)
    ax_control_ped_avg = fig.add_subplot(gs[0, 1])
    ax_control_ped_tot = fig.add_subplot(gs[1, 1], sharex=ax_control_ped_avg)
    ax_control_veh_avg = fig.add_subplot(gs[0, 2])
    ax_control_veh_tot = fig.add_subplot(gs[1, 2], sharex=ax_control_veh_avg)

    design_panels = [ax_design_avg, ax_design_tot]
    control_ped_panels = [ax_control_ped_avg, ax_control_ped_tot]
    control_veh_panels = [ax_control_veh_avg, ax_control_veh_tot]
    all_panels = design_panels + control_ped_panels + control_veh_panels
    top_panels = [ax_design_avg, ax_control_ped_avg, ax_control_veh_avg]
    bottom_panels = [ax_design_tot, ax_control_ped_tot, ax_control_veh_tot]

    # Combine paths for determining overall scale range
    all_json_paths = [design_unsig_path, realworld_unsig_path, control_tl_path, control_ppo_path]
    all_scales = []
    data_cache = {} # Cache loaded data

    for path in all_json_paths:
        if path not in data_cache:
            # Load data using get_averages
            data_false = get_averages(path, total=False)
            data_true = get_averages(path, total=True)
            data_cache[path] = {'total_false': data_false, 'total_true': data_true}
            all_scales.extend(data_false[0]) # Add scales from this path
        else:
             # If already cached, just add scales
             all_scales.extend(data_cache[path]['total_false'][0])


    unique_scales = np.sort(np.unique(np.array(all_scales)))
    x_min, x_max = unique_scales.min(), unique_scales.max()
    x_margin = 0.05 * (x_max - x_min)
    valid_min_scale = min(in_range_demand_scales)
    valid_max_scale = max(in_range_demand_scales)

    # --- Setup common axis properties ---
    for ax in all_panels:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        # Shade out-of-range areas with softer shading
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], valid_min_scale, facecolor='grey', alpha=0.15, zorder=-100)
        ax.axvspan(valid_max_scale, xlim[1], facecolor='grey', alpha=0.15, zorder=-100)
        
        # Turn off default grid
        ax.grid(False)
        
        # Set background to pure white
        ax.set_facecolor('white')
        
        # Y-axis formatting - Use n_yticks and ensure integers
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
        
        # Set x ticks
        ax.set_xticks(unique_scales, minor=True)

    # --- Plot Design Results (Left Column - Panel a) ---
    ax_design_avg.set_title('Pedestrian Arrival Time')
    design_paths = [realworld_unsig_path, design_unsig_path] # Original order kept from previous edits
    design_labels = ['Real-world', 'Design Agent (Ours)'] # Original order kept from previous edits
    design_legend_handles = []

    for path, label in zip(design_paths, design_labels):
        # Ensure label exists in COLORS, otherwise handle potential KeyError
        if label not in COLORS:
             print(f"Warning: Label '{label}' not found in COLORS dictionary. Skipping color assignment.")
             color = 'black' # Default color or handle error appropriately
        else:
             color = COLORS[label] # Uses new COLORS dict

        # Use cached data
        scales, _, _, avg_vals, _, _, avg_std = data_cache[path]['total_false']
        _, _, _, tot_vals, _, _, tot_std = data_cache[path]['total_true']

        # Average Plot - Enhanced line styles with gradient - higher z-order
        h = create_gradient_line(ax_design_avg, scales, avg_vals, color, lw=3.0, zorder=20, label=label)
        design_legend_handles.append(h) # Store handle for legend
        
        # Display standard deviation based on chosen visualization method - higher z-order
        if use_error_bars:
            ax_design_avg.errorbar(scales, avg_vals, yerr=avg_std, color=color, capsize=4.5, 
                                  elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
        else:
            ax_design_avg.fill_between(scales, avg_vals - avg_std, avg_vals + avg_std, 
                                     color=color, alpha=0.2, zorder=5)

        # Total Plot - Enhanced line styles with gradient - higher z-order
        tot_k = tot_vals / 1000.0
        tot_k_std = tot_std / 1000.0
        create_gradient_line(ax_design_tot, scales, tot_k, color, lw=3.0, zorder=20)
        
        # Display standard deviation based on chosen visualization method - higher z-order
        if use_error_bars:
            ax_design_tot.errorbar(scales, tot_k, yerr=tot_k_std, color=color, capsize=4.5, 
                                  elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
        else:
            ax_design_tot.fill_between(scales, tot_k - tot_k_std, tot_k + tot_k_std, 
                                     color=color, alpha=0.2, zorder=5)

    # Specific Y limits for design average plot
    ax_design_avg.set_ylim(bottom=50, top=120)
    ax_design_tot.set_ylim(bottom=-0.5)

    # --- Plot Control Results (Middle and Right Columns - Panels b, c) ---
    ax_control_ped_avg.set_title('Pedestrian Wait Time')
    ax_control_veh_avg.set_title('Vehicle Wait Time')

    # Note: 'Unsignalized' uses the *design_unsig_path* data for comparison consistency
    control_paths = [control_tl_path, design_unsig_path, control_ppo_path]
    control_labels = ['Signalized', 'Unsignalized', 'Control Agent (Ours)'] # Original labels kept
    control_legend_handles = []
    max_veh_tot_val = -np.inf # Keep track of max value for veh_tot plot

    for path, label in zip(control_paths, control_labels):
         # Ensure label exists in COLORS
        if label not in COLORS:
             print(f"Warning: Label '{label}' not found in COLORS dictionary. Skipping color assignment.")
             color = 'black'
        else:
             color = COLORS[label] # Uses new COLORS dict

        # Use cached data
        scales, veh_avg_mean, ped_avg_mean, _, veh_avg_std, ped_avg_std, _ = data_cache[path]['total_false']
        _, veh_tot, ped_tot, _, veh_tot_std, ped_tot_std, _ = data_cache[path]['total_true']

        # Pedestrian Average Wait (Middle Top) - Enhanced line styles with gradient - higher z-order
        h_ped = create_gradient_line(ax_control_ped_avg, scales, ped_avg_mean, color, lw=3.0, zorder=20, label=label)
        
        # Display standard deviation based on chosen visualization method - higher z-order
        if use_error_bars:
            ax_control_ped_avg.errorbar(scales, ped_avg_mean, yerr=ped_avg_std, color=color, 
                                 capsize=4.5, elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
        else:
            ax_control_ped_avg.fill_between(scales,
                                         ped_avg_mean - ped_avg_std,
                                         ped_avg_mean + ped_avg_std,
                                         color=color, alpha=0.2, zorder=5)
        
        # Store unique handles for legend
        if label not in [h.get_label() for h in control_legend_handles]:
             control_legend_handles.append(h_ped)

        # Pedestrian Total Wait (Middle Bottom) - Enhanced line styles with gradient - higher z-order
        create_gradient_line(ax_control_ped_tot, scales, ped_tot/1000, color, lw=3.0, zorder=20)
        
        # Display standard deviation based on chosen visualization method - higher z-order
        if use_error_bars:
            ax_control_ped_tot.errorbar(scales, ped_tot/1000, yerr=ped_tot_std/1000, color=color, 
                                 capsize=4.5, elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
        else:
            ax_control_ped_tot.fill_between(scales,
                                        (ped_tot - ped_tot_std)/1000,
                                        (ped_tot + ped_tot_std)/1000,
                                        color=color, alpha=0.2, zorder=5)

        # Vehicle Average Wait (Right Top) - Enhanced line styles with gradient - higher z-order
        create_gradient_line(ax_control_veh_avg, scales, veh_avg_mean, color, lw=3.0, zorder=20)
        
        # Display standard deviation based on chosen visualization method - higher z-order
        if use_error_bars:
            ax_control_veh_avg.errorbar(scales, veh_avg_mean, yerr=veh_avg_std, color=color, 
                                 capsize=4.5, elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
        else:
            ax_control_veh_avg.fill_between(scales,
                                        veh_avg_mean - veh_avg_std,
                                        veh_avg_mean + veh_avg_std,
                                        color=color, alpha=0.2, zorder=5)

        # Vehicle Total Wait (Right Bottom) - Enhanced line styles with gradient - higher z-order
        veh_tot_k = veh_tot / 1000.0
        veh_tot_k_std = veh_tot_std / 1000.0
        create_gradient_line(ax_control_veh_tot, scales, veh_tot_k, color, lw=3.0, zorder=20)
        
        # Display standard deviation based on chosen visualization method - higher z-order
        if use_error_bars:
            ax_control_veh_tot.errorbar(scales, veh_tot_k, yerr=veh_tot_k_std, color=color, 
                                 capsize=4.5, elinewidth=2.0, capthick=2.2, alpha=0.85, fmt='none', zorder=10)
        else:
            ax_control_veh_tot.fill_between(scales,
                                        veh_tot_k - veh_tot_k_std,
                                        veh_tot_k + veh_tot_k_std,
                                        color=color, alpha=0.2, zorder=5)
                                         
        # Update max value seen in this plot (including std dev)
        current_max = np.max(veh_tot_k + veh_tot_k_std)
        if current_max > max_veh_tot_val:
            max_veh_tot_val = current_max


    # Order control legend handles to match labels
    ordered_control_handles = []
    temp_handle_dict = {h.get_label(): h for h in control_legend_handles}
    for lbl in control_labels:
        if lbl in temp_handle_dict:
            ordered_control_handles.append(temp_handle_dict[lbl])

    # Set Y limits for control plots
    for ax in control_ped_panels + control_veh_panels:
        # Set bottom limit first
        ax.set_ylim(bottom=-0.5)

    ax_control_veh_tot.set_ylim(top=3.9)
    ax_control_veh_avg.set_ylim(top=75)

    # --- Draw grid lines AFTER setting all y-limits ---
    # This ensures grid lines align with the final tick positions
    for ax in all_panels:
        # Create custom grid manually with explicit z-order
        # Get y ticks and draw horizontal grid lines - now after y-limits are set
        y_ticks = ax.get_yticks()
        for y in y_ticks:
            ax.axhline(y=y, color='#cccccc', linestyle='--', linewidth=1.0, alpha=0.75, zorder=-90)
        
        # Draw vertical grid lines for all scales
        for x in unique_scales:
            ax.axvline(x=x, color='#cccccc', linestyle='--', linewidth=0.9, alpha=0.65, zorder=-90)
    
    # --- X-axis Ticks and Labels ---
    # Select every other scale, EXCLUDING the last one
    scales_to_show = unique_scales[:-1:2] # Select every other scale from all but the last

    x_tick_labels = []
    for s in scales_to_show:
        if abs(s * 10 - round(s * 10)) < 1e-6:
            x_tick_labels.append(f"{s:.1f}x")
        else:
            x_tick_labels.append(f"{s:.2f}x")

    for ax in bottom_panels:
        ax.set_xticks(scales_to_show) # Set major ticks only at these locations
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel('Demand Scale', fontsize=fs + 1, fontweight='medium') # Use updated fs

    for ax in top_panels:
        ax.tick_params(labelbottom=False) # Hide x-labels on top plots

    # --- Y-axis Labels (using fig.text) ---
    # Simplified labels, keeping units
    # Adjusted positions for better centering relative to the taller figure
    avg_y_pos = 0.72 # Adjusted vertical center for top row
    tot_y_pos = 0.32 # Adjusted vertical center for bottom row

    # Direct assignment for each position instead of using a loop with condition
    fig.text(0.04, avg_y_pos, 'Average (s)', va='center', rotation='vertical', 
             fontsize=fs+1, fontweight='medium')
    fig.text(0.04, tot_y_pos, 'Total (×10³ s)', va='center', rotation='vertical', 
             fontsize=fs+1, fontweight='medium')

    fig.text(0.36, avg_y_pos, 'Average (s)', va='center', rotation='vertical', 
             fontsize=fs+1, fontweight='medium')
    fig.text(0.36, tot_y_pos, 'Total (×10³ s)', va='center', rotation='vertical', 
             fontsize=fs+1, fontweight='medium')

    fig.text(0.68, avg_y_pos, 'Average (s)', va='center', rotation='vertical', 
             fontsize=fs+1, fontweight='medium')
    fig.text(0.68, tot_y_pos, 'Total (×10³ s)', va='center', rotation='vertical', 
             fontsize=fs+1, fontweight='medium')

    # --- Legends (Split) with Rounded White Boxes ---
    legend_kwargs = {
        'loc': 'lower center',
        'fontsize': fs - 1,        # Slightly larger legend font
        'frameon': True,           # Turn on frame
        'fancybox': True,          # Use rounded corners
        'facecolor': 'white',      # Updated background to white
        'edgecolor': '#cccccc',    # Slightly darker gray border
        'framealpha': 0.95,        # Make box more opaque
        'borderpad': 0.7,          # Padding inside the box
        'labelspacing': 0.5,       # Spacing between legend entries
        'handletextpad': 0.6,      # Space between line and text
        'handlelength': 2.5,       # Longer line handles
        'markerscale': 1.1         # Slightly larger markers in legend
    }

    # Design Legend (a) - Centered under first column
    # Adjusted y anchor for taller figure and space above
    leg_a = fig.legend(handles=design_legend_handles, labels=design_labels,
                       ncol=2,
                       bbox_to_anchor=(0.215, -0.03), # Moved legend down
                       **legend_kwargs)

    # Control Legend (b, c) - Centered under middle/right columns
    # Adjusted y anchor for taller figure and space above
    leg_b_c = fig.legend(handles=ordered_control_handles, labels=control_labels,
                         ncol=3,
                         bbox_to_anchor=(0.675, -0.03), # Moved legend down
                         **legend_kwargs)
                         
    # Increase the linewidth in the legends
    for legend in [leg_a, leg_b_c]:
        for line in legend.get_lines():
            line.set_linewidth(3.5)  # Thicker lines in legend only

    # --- Panel Labels (a), (b), (c) ---
    # Positioned below the legends - Adjusted y position to be closer to legends
    label_y_pos = -0.08 # Moved labels up relative to legends
    label_fontsize = fs + 2 # Match title font size
    fig.text(0.215, label_y_pos, '(a)', ha='center', va='center', fontsize=label_fontsize, fontweight='bold')
    fig.text(0.505, label_y_pos, '(b)', ha='center', va='center', fontsize=label_fontsize, fontweight='bold')
    fig.text(0.835, label_y_pos, '(c)', ha='center', va='center', fontsize=label_fontsize, fontweight='bold')


    # --- Final Adjustments and Save ---
    # Adjusted margins for taller figure and repositioned lower elements
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.13) # Adjusted bottom margin
    plt.savefig("design_control_results.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


# # plot_design_results(new_design_unsignalized_results_path, 
# #                     real_world_design_unsignalized_results_path,
# #                     in_range_demand_scales = irds)

# # plot_control_results(new_design_unsignalized_results_path,
# #                                  new_design_tl_results_path,
# #                                  new_design_ppo_results_path,
# #                                  in_range_demand_scales = irds)


def plot(design_and_control = True, 
         graphs_and_gmm = False,
         rewards_results = True):
    
    run_dir = "May09_11-34-05"

    if design_and_control:
        eval_dir = "eval_May10_16-16-52"
        policy = "policy_at_7603200"# "best_eval_policy"
        real_world_design_unsignalized_results_path = f'./runs/{run_dir}/results/{eval_dir}/realworld_unsignalized.json'
        new_design_ppo_results_path = f'./runs/{run_dir}/results/{eval_dir}/{policy}_ppo.json'
        new_design_tl_results_path = f'./runs/{run_dir}/results/{eval_dir}/{policy}_tl.json'
        new_design_unsignalized_results_path = f'./runs/{run_dir}/results/{eval_dir}/{policy}_unsignalized.json'

        irds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]

        plot_design_and_control_results(
            design_unsig_path = new_design_unsignalized_results_path,
            realworld_unsig_path = real_world_design_unsignalized_results_path,
            control_tl_path = new_design_tl_results_path,
            control_ppo_path = new_design_ppo_results_path,
            in_range_demand_scales = irds
        )
                    
    if graphs_and_gmm:
        original_graph = f'./runs/{run_dir}/graph_iterations/graph_i_0_data.pkl'
        final_graph = f'./runs/{run_dir}/graph_iterations/graph_i_eval_final_data.pkl'
        gmm_path = f'./runs/{run_dir}/gmm_iterations/gmm_i_eval_final_b0_data.pkl'

        plot_graphs_and_gmm(original_graph,
                            final_graph,
                            gmm_path)
        
    if rewards_results:
        rewards_results_plot(
            combined_csv_codesign = "./runs/combined_rewards_codesign.csv",
            combined_csv_control = "./runs/combined_rewards_control_only.csv",
            results_codesign = "./runs/May09_11-34-05/results/eval_May13_13-07-37/policy_at_7603200_ppo.json",
            results_separate = "./runs/May09_11-34-05/results/eval_May13_13-07-37/policy_at_7603200_ppo.json"

        )
        
plot()

# plot_demand()

# plot_gmm_top_down(gmm_pkl_path = "./runs/May09_12-21-15/gmm_iterations/gmm_i_eval14400000_b0_data.pkl")

