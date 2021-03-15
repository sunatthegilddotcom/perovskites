# -*- coding: utf-8 -*-
"""
This function contains visualization functions for the linear regression.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# DEFAULT MIN-MAX VALUES for ENVIRONMENTAL CONDITIONS
NSUNS_RANGE = np.array([1, 32])
REL_HUM_RANGE = np.array([0, 60])
TEMP_RANGE = np.array([25, 85])
MA_RANGE = np.array([0, 1])
STYLE_VARIABLES = ['T', 'RH', 'MA', 'O2', 'soakSuns']

# default font
default_font = {'color': 'k',
                'fontsize': 14,
                'fontfamily': 'Arial'
                }

# Append the current folder to sys path
curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
sys.path.append(curr_dir)
import image_processing as impr

###############################################################################
# SCATTER PLOT MARKER STYLING BASED ON ENV. CONDITIONS
###############################################################################
def error_evolution_plot(ax, error_list, x_list):
    ax.scatter(x_list, error_list, s=8)
    ax.scatter(x_list, error_list, s=4, c='w')
    ax.plot(x_list, error_list, linewidth=1.5)


def coefficient_bar_chart(ax, feat_labels, coeff_values, tol=1e-4):
    """
    Makes the coefficient bar chart for the trained linear regression model.

    Parameters
    ----------
    ax : Matplotlib.Axes object()
        The Matplotlib.Axes object().
    feat_labels : seq
        The feature labels.
    coeff_values : numpy.ndarray
        The corresponding features weights.
    tol : float, optional
        The maximum weight value to display. The default is 1e-4.

    Raises
    ------
    ValueError
        When feat_labels and coeff_values are not of same length.

    Returns
    -------
    Matplotlib scatter plot handle

    """
    coeff_values = np.array(coeff_values)
    if len(feat_labels) != len(coeff_values):
        raise ValueError("The feats_labels and X's columns must be of same\n\
                         length.")
    feat_labels = np.array(feat_labels)[np.abs(coeff_values) > tol]
    coeff_values = coeff_values[np.abs(coeff_values) > tol]
    feat_labels = feat_labels[np.argsort(np.abs(coeff_values))[::-1]]
    coeff_values = coeff_values[np.argsort(np.abs(coeff_values))[::-1]]

    sign_colors = ['r' if i <= 0 else 'g' for i in coeff_values]
    bar_x = np.arange(len(feat_labels))
    ax.set_title('Top {} selected coeff.'.format(len(feat_labels)),
                 **default_font)
    ax.set_ylabel('Abs. value of coefficient', **default_font)
    ax.set_xlabel('Feature name', **default_font)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(feat_labels, horizontalalignment='right',
                       fontsize=int(default_font['fontsize']*0.8))
    ax.tick_params(axis='x', labelrotation=45)
    ax.bar(bar_x, np.abs(coeff_values), color=sign_colors)

    handle = ax.scatter(bar_x[-1],
                        np.max(np.abs(coeff_values)),
                        c='w', s=1)
    return handle


def parity_plot(ax, Y_true, Y_pred, y_label, style_args=[]):
    """Makes the Parity plot in the matplotlib.Axes() object given.
    For use, first make an Axes(). For example :
        fig, ax = plt.subplots(1,1)
    Then pass this ax into the function.

    Paramters
    ---------------
    ax : matplotlib.Axes() object
        The axes object to make the plot
    Y_true : np.array
        True Y values
    Y_pred : np.array
        Predicted Y values
    y_label : str
        The name of y
    style_args : list of dicts, optional
        List of marker stlying dicts. Each element in list
        must be a styling dict for corresponding data point.

    Returns
    ------------------
    Matplotlib scatter plot handle

    """

    if y_label is not None:
        ax.set_ylabel('$' + y_label + '_{pred}$', **default_font)
        ax.set_xlabel('$' + y_label + '_{obs}$', **default_font)
    ax.tick_params(axis='x', labelsize=default_font['fontsize'])
    ax.tick_params(axis='y', labelsize=default_font['fontsize'])

    max_point = np.max([np.max(Y_true), np.max(Y_pred)])
    min_point = np.min([np.min(Y_true), np.min(Y_pred)])

    ax.plot([min_point, max_point], [min_point, max_point],
            linestyle='dashed',
            color='gray',
            linewidth=1.5,
            alpha=0.75
            )

    alpha = 0.85
    if len(style_args) == 0:
        handle = ax.scatter(Y_true, Y_pred, alpha=alpha)
    else:
        for i in range(len(Y_true)):
            handle = ax.scatter(Y_true[i], Y_pred[i],
                                **(style_args[i]), alpha=alpha)

    return handle


def default_style_legend(save_path=None, dpi=100, font=default_font):
    """
    Saves a figure to define the color coding in scatter parity plots. This
    figure can be used as a legend to describe color coding. This generates
    the new marker style legend automatically when changes are made to
    the style_coding function.

    Paramters
    -------------
    save_path : str, optional
        path to save the new legend plot. The default is None.
    dpi : int, optional
        The quality of image in dpi. The default is 100
    font : dict, optional
        The font styling dictionary. The default is default_font.

    Returns
    -------------
    None

    """

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # creating dummy data
    x = 4 + np.arange(4)
    y = 4 + np.arange(4)
    X, Y = np.meshgrid(x, y)

    x_values = [0, 20, 40, 60]
    y_values = [1, 8, 16, 32]
    X_val, Y_val = np.meshgrid(x_values, y_values[::-1])

    # Defining Relative Humidity and Sun intensity coding style --------------
    # Get the marker styles
    dict_list = style_coding(T=np.zeros(len(X_val.flatten())) + 25,
                             RH=X_val.flatten(),
                             Nsuns=Y_val.flatten(),
                             MA=np.ones(len(X_val.flatten())))

    for i in range(len(X.flatten())):
        ax.scatter(X.flatten()[i], Y.flatten()[i], **(dict_list[i]))

    xlabels = ['0%', '20%', '40%', '60%']
    ylabels = ['1 Sun', '8 Suns', '16 Suns', '32 Suns']
    for i in range(len(xlabels)):
        ax.text(s=xlabels[i], x=x[i], y=y[-1]+0.35, **font, ha='center')
    for i in range(len(ylabels)):
        ax.text(s=ylabels[i], x=x[0]-1.45, y=y[::-1][i], **font, va='center')

    ax.text(s="Relative Humidity", x=0.5*(x[0]+x[1]), y=y[-1]+1, **font)
    ax.text(s="Sun Intensity", y=y[1], x=x[0]-1.87,
            rotation=90, **font, ha='center')
    ax.arrow(x[0], y[-1]+0.80, 3, 0, head_width=0.1, head_length=0.1,
             fc='k', ec='k')
    ax.arrow(x[0]-1.6, y[-1], 0, -3, head_width=0.1, head_length=0.1,
             fc='k', ec='k')

    # Defining MA% coding of style -------------------------------------------
    x2_labels = ['0%\nMA', '50%\nMA', '100%\nMA']
    x2 = np.array([x[0]*0.85 + 0.15*x[1],
                   0.5*(x[1] + x[2]),
                   x[2]*0.15 + 0.85*x[3]])
    x2 = x2 - 1.2
    # Get the marker styles
    dict_list = style_coding(T=np.zeros(len(x2)) + 25,
                             RH=np.zeros(len(x2))+20,
                             Nsuns=np.zeros(len(x2))+8,
                             MA=[0, 0.5, 1],
                             )
    for i in range(len(x2)):
        ax.scatter(x2[i], y[0]-1, **(dict_list[i]))

    for i in range(len(x2_labels)):
        ax.text(s=x2_labels[i], x=x2[i], y=y[0]-1.75, **font, ha='center')

    # Defining Temperature coding of style -----------------------------------
    x3_labels = ['$25^o$ C', '$45^o$ C', '$65^o$ C', '$85^o$ C']
    # Get the marker styles
    dict_list = style_coding(T=[25, 45, 65, 85],
                             RH=np.zeros(len(x3_labels))+20,
                             Nsuns=np.zeros(len(x3_labels))+8,
                             MA=np.ones(len(x3_labels)))
    for i in range(len(x3_labels)):
        ax.scatter(x[-1]+1.15, y[::-1][i], **(dict_list[i]))
    for i in range(len(x3_labels)):
        ax.text(s=x3_labels[i], x=x[-1]+1.45, y=y[::-1][i],
                **font, va='center')
    ax.arrow(x[-1]+2.6, y[-1], 0, -3, head_width=0.1, head_length=0.1,
             fc='k', ec='k')
    ax.text(s="Temperature", y=y[1], x=x[-1]+2.8,
            rotation=-90, **font, ha='center')

    # Defining environmental medium coding of style ---------------------------
    x4_labels = ['$N_2$', 'air', '$O_2$']
    x4 = np.array([x[0]*0.2 + 0.8*x[1],
                   0.5*(x[1] + x[2]),
                   x[2]*0.8 + 0.2*x[3]])
    x4 = x4 + 2.8
    # Get the marker styles
    dict_list = style_coding(T=np.zeros(len(x4))+25,
                             RH=np.zeros(len(x4))+20,
                             Nsuns=np.zeros(len(x4))+8,
                             MA=np.ones(len(x4)),
                             O2=np.array([0, 21, 100]))
    for i in range(len(x4)):
        ax.scatter(x4[i], y[0]-1, **(dict_list[i]))
    for i in range(len(x4_labels)):
        ax.text(s=x4_labels[i], x=x4[i], y=y[0]-1.5, **font, ha='center')

    ax.set_ylim([1.85, 8.5])
    ax.set_xlim([1.85, 10])
    ax.axis('off')

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        return fig


def style_coding(T=None, RH=None, Nsuns=None, MA=None,
                 O2=None, N2=None, use_default_minmax=True):
    """Color coding to be used for scatter plots based on environmental
    conditions. This function returns the list of style dicts to be used
    directly in the scatter plot. Each dict in list corresponds to the marker
    style for a data point.

    Paramters
    -----------------
    T : numpy.array or None
        Temperature in C
    RH : numpy.array or None
        Relative Humidity in %
    Nsuns : numpy.array or None
        Number of suns intensity
    MA : numpy.array or None
        MA feature value in frac.
    O2 : numpy.array or None
        % of O2
    N2 : numpy.array or None
        % of N2
    use_default_minmax : bool
        If True, then the default values listed in the beginning
        of the file will be used for normalizing the values. Otherwise,
        the min and maximum of the input arguement arrays will be used

    Returns
    -----------------
    list
        The list of dictionaries corresponding to the marker styles of each
        data point

    """

    cmap_edge = mpl.cm.get_cmap('jet')
    cmap_fill = mpl.cm.get_cmap('Greys')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # Define which minimum and maximum values to use for scaling
    if use_default_minmax:
        minmax_T = TEMP_RANGE
        minmax_RH = REL_HUM_RANGE
        minmax_Nsuns = NSUNS_RANGE
        minmax_MA = MA_RANGE
    else:
        minmax_T = None
        minmax_RH = None
        minmax_Nsuns = None
        minmax_MA = None

    # Encapsulation determined by fill color
    if MA is not None:
        MA_norm = impr.normalize(1-np.array(MA), pix_range=(0, 0.8),
                                 minmax=minmax_MA)
        c = norm([cmap_fill(i) for i in MA_norm])
    else:
        c = 'w'

    # edgecolor determinined by temperature
    if T is not None:
        T_norm = impr.normalize(T, minmax=minmax_T)
        edgecolor = norm([cmap_edge(i) for i in T_norm])
    else:
        edgecolor = 'k'

    # Nsuns determined by marker size
    if Nsuns is not None:
        Nsuns_norm = impr.normalize(np.log(1 + Nsuns),
                                    minmax=np.log(minmax_Nsuns+1))
        size = 100 + Nsuns_norm*350
    else:
        size = 100

    # edge_width and thus fill determined by relative humidity
    # change size and linewidth in such a way that the overall marker size
    # stays constant
    if RH is not None:
        RH_norm = impr.normalize(RH, pix_range=(0.1, 0.6),
                                 minmax=minmax_RH)

        def props_for_RH(RH_norm):
            # The following func. returns linewidth to fully fill the marker,
            # but size changes
            p1 = np.array([2.14415400e-08, -4.77467295e-05,
                           5.26042769e-02, 4.70106892e+00])

            def linewidth_for_fullfill(x):
                return np.matmul(p1, [x**n for n in np.arange(len(p1)-1,
                                                              -1, -1)])

            # The following func. returns size of marker for full fill when
            # used with above function for linewidth
            p2 = np.array([-9.33333333e-08, 1.03736264e-04,
                           2.67263004e-01, -7.66153846e+00])

            def same_size_fullfill(x):
                return np.matmul(p2, [x**n for n in np.arange(len(p2)-1,
                                                              -1, -1)])

            curr_size = size + RH_norm*(same_size_fullfill(size)-size)
            line_width = linewidth_for_fullfill(same_size_fullfill(size))
            line_width = RH_norm*line_width
            return curr_size, line_width

        s, linewidth = props_for_RH(RH_norm)
    else:
        s, linewidth = props_for_RH(1)

    # The marker style is set by the gas conditions
    gas = None
    if not (O2 is None):
        gas = O2.copy()
        gas_type = 'O2'
    elif not (N2 is None):
        gas = N2.copy()
        gas_type = 'N2'

    if not (gas is None):
        if gas_type == 'N2':
            air_ind = np.logical_and(gas < 80, gas > 70)
            N2_ind = (gas == 100)
            O2_ind = (gas == 0)

        elif gas_type == 'O2':
            air_ind = np.logical_and(gas < 30, gas > 20)
            N2_ind = (gas == 0)
            O2_ind = (gas == 100)

        gas = gas.astype('<U1')
        gas[air_ind] = 'o'  # air
        gas[N2_ind] = '^'   # N2
        gas[O2_ind] = 'v'   # O2
        gas_markers = gas
    else:
        gas_markers = ["o" for i in range(len(T))]

    dict_list = []
    for i in range(len(T)):
        dict_list += [{
                    'c': c[i],
                    's': s[i],
                    'linewidths': linewidth[i],
                    'edgecolors': edgecolor[i],
                    'marker': gas_markers[i]
        }]

    return dict_list
