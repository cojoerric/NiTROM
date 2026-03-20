import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from pathlib import Path


def save_uncover_line_slides(
    x,
    y_series,
    labels,
    output_prefix,
    *,
    colors=None,
    linestyles=None,
    linewidths=None,
    xlabel='Time $t$',
    ylabel='$y(t)$',
    xlim=None,
    ylim=None,
    legend_loc='upper left',
    fixed_legend_size=True,
    show_legend=True,
    log_y=False,
    callback=None,
):
    if len(y_series) != len(labels):
        raise ValueError("'y_series' and 'labels' must have the same length.")

    n_lines = len(y_series)
    if colors is None:
        colors = [None] * n_lines
    if linestyles is None:
        linestyles = [None] * n_lines
    if linewidths is None:
        linewidths = [2.0] * n_lines

    if not (len(colors) == len(linestyles) == len(linewidths) == n_lines):
        raise ValueError("Style lists must match the number of lines.")

    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major', color='#d7d7d7', linewidth=0.55, alpha=0.7)
    ax.grid(which='minor', color='#efefef', linewidth=0.4, alpha=0.8)

    if xlim is not None:
        ax.set_xlim(xlim)

    lines = []
    for y, label, color, linestyle, linewidth in zip(y_series, labels, colors, linestyles, linewidths):
        line = ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)[0]
        line.set_visible(False)
        lines.append(line)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        all_y = torch.cat([torch.as_tensor(y).reshape(-1) for y in y_series])
        if log_y and torch.any(all_y <= 0):
            raise ValueError("All y-values must be positive when 'log_y=True' and 'ylim' is not provided.")
        ax.set_ylim(all_y.min().item(), all_y.max().item())

    legend = None
    proxy_hidden = Line2D([], [], linestyle='None', linewidth=0, alpha=0)

    ax.axvspan(0.0, 10.0, color='#ececec', alpha=0.9, zorder=0)
    ax.text(
        0.16, 0.88, 'Training window',
        transform=ax.transAxes,
        ha='center', va='center',
        fontsize=11,
        bbox=dict(facecolor='white', edgecolor='#6f6f6f', linewidth=0.6, alpha=0.95, boxstyle='round,pad=0.25')
    )

    fig.tight_layout()
    for i, line in enumerate(lines, 1):
        line.set_visible(True)

        if show_legend:
            if legend is not None:
                legend.remove()

            if fixed_legend_size:
                legend_handles = []
                legend_labels = []
                for j, current_line in enumerate(lines):
                    if j < i:
                        legend_handles.append(current_line)
                        legend_labels.append(current_line.get_label())
                    else:
                        legend_handles.append(proxy_hidden)
                        legend_labels.append(' ')
                legend = ax.legend(legend_handles, legend_labels, loc=legend_loc, handlelength=2.0)
                for txt, alpha in zip(legend.get_texts(), [1.0] * i + [0.0] * (n_lines - i)):
                    txt.set_alpha(alpha)
            else:
                legend = ax.legend(lines[:i], labels[:i], loc=legend_loc)

        fig.canvas.draw()
        fig.savefig(f'{output_prefix}_{i}.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
        fig.savefig(f'{output_prefix}_{i}.eps', format='eps')
        fig.savefig(f'{output_prefix}_{i}.pdf', format='pdf')

    plt.close(fig)
