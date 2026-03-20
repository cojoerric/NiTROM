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
    window_shading=False,
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

    if window_shading:
        ax.axvspan(0.0, 20.0, color='#ececec', alpha=0.9, zorder=0)
        ax.text(
            0.26, 0.12, 'Training window',
            transform=ax.transAxes,
            ha='center', va='bottom',
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


def save_uncover_multiaxes_slides(
    x,
    y_groups,
    labels,
    output_prefix,
    *,
    colors=None,
    linestyles=None,
    linewidths=None,
    ncols=1,
    figsize=None,
    xlabels=None,
    ylabels=None,
    titles=None,
    xlims=None,
    ylims=None,
    texts=None,
    hide_xticks_except_last=False,
    legend_axis_index=0,
    legend_loc='upper left',
    fixed_legend_size=True,
    show_legend=True,
    log_y=False,
    save_eps=False,
):
    n_axes = len(y_groups)
    n_lines = len(labels)

    if n_axes == 0:
        raise ValueError("'y_groups' must contain at least one subplot.")

    for group in y_groups:
        if len(group) != n_lines:
            raise ValueError("Each entry in 'y_groups' must have the same number of series as 'labels'.")

    if colors is None:
        colors = [None] * n_lines
    if linestyles is None:
        linestyles = [None] * n_lines
    if linewidths is None:
        linewidths = [2.0] * n_lines

    if not (len(colors) == len(linestyles) == len(linewidths) == n_lines):
        raise ValueError("Style lists must match the number of labels.")

    def _expand(values):
        if values is None:
            return [None] * n_axes
        if len(values) != n_axes:
            raise ValueError("Per-axis option length must match the number of subplots.")
        return values

    xlabels = _expand(xlabels)
    ylabels = _expand(ylabels)
    titles = _expand(titles)
    xlims = _expand(xlims)
    ylims = _expand(ylims)
    texts = _expand(texts)

    output_path = Path(output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nrows = (n_axes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    lines_by_axis = []
    for axis_index, ax in enumerate(axes_flat):
        if axis_index >= n_axes:
            ax.axis('off')
            continue

        if xlabels[axis_index] is not None:
            ax.set_xlabel(xlabels[axis_index])
        if ylabels[axis_index] is not None:
            ax.set_ylabel(ylabels[axis_index])
        if titles[axis_index] is not None:
            ax.set_title(titles[axis_index])
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
        if xlims[axis_index] is not None:
            ax.set_xlim(xlims[axis_index])

        subplot_lines = []
        for y, label, color, linestyle, linewidth in zip(
            y_groups[axis_index], labels, colors, linestyles, linewidths
        ):
            line = ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)[0]
            line.set_visible(False)
            subplot_lines.append(line)
        lines_by_axis.append(subplot_lines)

        if ylims[axis_index] is not None:
            ax.set_ylim(ylims[axis_index])

        if hide_xticks_except_last and axis_index < n_axes - 1:
            ax.set_xticks([])

        if texts[axis_index] is not None:
            text_spec = texts[axis_index]
            ax.text(
                text_spec.get('x', 0.5),
                text_spec.get('y', 0.5),
                text_spec.get('text', ''),
                transform=ax.transAxes,
                ha=text_spec.get('ha', 'center'),
                va=text_spec.get('va', 'bottom'),
                fontsize=text_spec.get('fontsize', 16),
            )

    legend = None
    proxy_hidden = Line2D([], [], linestyle='None', linewidth=0, alpha=0)
    legend_ax = axes_flat[legend_axis_index]

    fig.tight_layout()
    for reveal_index in range(1, n_lines + 1):
        for subplot_lines in lines_by_axis:
            subplot_lines[reveal_index - 1].set_visible(True)

        if show_legend:
            if legend is not None:
                legend.remove()

            reference_lines = lines_by_axis[legend_axis_index]
            if fixed_legend_size:
                legend_handles = []
                legend_labels = []
                for j, current_line in enumerate(reference_lines):
                    if j < reveal_index:
                        legend_handles.append(current_line)
                        legend_labels.append(current_line.get_label())
                    else:
                        legend_handles.append(proxy_hidden)
                        legend_labels.append(' ')
                legend = legend_ax.legend(legend_handles, legend_labels, loc=legend_loc, handlelength=2.0)
                for txt, alpha in zip(legend.get_texts(), [1.0] * reveal_index + [0.0] * (n_lines - reveal_index)):
                    txt.set_alpha(alpha)
            else:
                legend = legend_ax.legend(reference_lines[:reveal_index], labels[:reveal_index], loc=legend_loc)

        fig.canvas.draw()
        fig.savefig(f'{output_prefix}_{reveal_index}.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
        if save_eps:
            fig.savefig(f'{output_prefix}_{reveal_index}.eps', format='eps')
        fig.savefig(f'{output_prefix}_{reveal_index}.pdf', format='pdf')

    plt.close(fig)
