# -*- coding: utf-8 -*-
"""
This function contains visualization functions for the linear regression.
"""

###############################################################################
# SCATTER PLOT MARKER STYLING BASED ON ENV. CONDITIONS
###############################################################################
    
def default_style_legend(font=default_font, save_path=SCATTER_LEGEND_PLOT):
    """
    Saves a figure to define the color coding.
    This figure can be used as a legend to describe color coding.
    
    This generates the new marker style legend automatically when changes are made to
    the style_coding function
    
    Paramters
    -------------
    font : dict (default : default_dict)
        The font styling dictionary
    save_path : str (default : SCATTER_LEGEND_PLOT)
        path to save the new legend plot
    
    Returns
    -------------
    None. Just makes the plots
    
    """
    
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    
    #creating dummy data
    x = 4 + np.arange(4)
    y = 4 + np.arange(4)
    X, Y = np.meshgrid(x, y)
    
    x_values = [0, 20, 40, 60]
    y_values = [1, 8, 16, 32]
    X_val, Y_val = np.meshgrid(x_values, y_values[::-1])
    
    # Defining Relative Humidity and Sun intensity codes ----------------
    # Get the marker styles
    dict_list = style_coding(T=np.zeros(len(X_val.flatten())) + 25,
                             RH=X_val.flatten(),
                             Nsuns=Y_val.flatten(),
                             Encaps=np.ones(len(X_val.flatten())))
    
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
    
    ax.arrow(x[0], y[-1]+0.80, 3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(x[0]-1.6, y[-1], 0, -3, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    ############################################################################
    ########### !!!!!!!! Encapsulation doding replaced with MA fraction
    ############################################################################
    
    # Defining Encapsulation codes ------------------------
    #x2_labels = ['No\nEncap.', '5 mg/mL\nPMMA', '10 mg/ml\nPMMA']
    x2_labels = ['0%\nMA', '50%\nMA', '100%\nMA']
    x2 = np.array([(x[0]*0.85 + 0.15*x[1]), 0.5*(x[1] + x[2]), (x[2]*0.15 + 0.85*x[3])])
    x2 = x2 - 1.2
    #Get the marker styles
    dict_list = style_coding(T=np.zeros(len(x2)) + 25,
                             RH=np.zeros(len(x2))+20,
                             Nsuns=np.zeros(len(x2))+8,
    #                         Encaps=[0,5,10],
                             Encaps=[0,0.5,1],
                             )
    for i in range(len(x2)):
        ax.scatter(x2[i], y[0]-1, **(dict_list[i]))
        
    for i in range(len(x2_labels)):
        ax.text(s=x2_labels[i], x=x2[i]-0.05, y=y[0]-1.75, **font, ha='center')
    
    # Defining Temperature Codes -----------------------------
    x3_labels = ['$25^o$ C', '$45^o$ C', '$65^o$ C', '$85^o$ C']
    #Get the marker styles
    dict_list = style_coding(T=[25,45,65,85],
                             RH=np.zeros(len(x3_labels))+20,
                             Nsuns=np.zeros(len(x3_labels))+8,
                             Encaps=np.ones(len(x3_labels)))
    for i in range(len(x3_labels)):
        ax.scatter(x[-1]+1.15, y[::-1][i], **(dict_list[i]))
    for i in range(len(x3_labels)):
        ax.text(s=x3_labels[i], x=x[-1]+1.45, y=y[::-1][i], **font, va='center')
    
    ax.arrow(x[-1]+2.4, y[-1], 0, -3, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(s="Temperature", y=y[1], x=x[-1]+2.6,
            rotation=-90, **font, ha='center')
    
    #Defining Gas codes ---------------------------------
    
    x4_labels = ['$N_2$', 'air', '$O_2$']
    x4 = np.array([(x[0]*0.2 + 0.8*x[1]), 0.5*(x[1] + x[2]), (x[2]*0.8 + 0.2*x[3])])
    x4 = x4 + 2.8
    #Get the marker styles
    dict_list = style_coding(T=np.zeros(len(x4))+ 25,
                             RH=np.zeros(len(x4))+20,
                             Nsuns=np.zeros(len(x4))+8,
                             Encaps=np.ones(len(x4)),
                             O2=np.array([0,21,100]))
    for i in range(len(x4)):
        ax.scatter(x4[i], y[0]-1, **(dict_list[i]))
        
    for i in range(len(x4_labels)):
        ax.text(s=x4_labels[i], x=x4[i], y=y[0]-1.5, **font, ha='center')
    
    
    
    ax.set_ylim([1.85, 8.5])
    ax.set_xlim([1.85, 10])
    ax.axis('off')
    
    if not save_path is None:
        fig.savefig(save_path)
        plt.close(fig)
        gc.collect()
 
    
def style_coding(T=None, RH=None, Nsuns=None, Encaps=None, O2=None, N2=None, use_default_minmax=True):
    """Color coding to be used for scatter plots based on environmental conditions.
    Returns the list of style dicts to be used directly in the scatter plot.
    Each dict in list corresponds to the marker style for a data point.
    
    Paramters
    -----------------
    T : numpy.array or None
        Temperature in C
    RH : numpy.array or None
        Relative Humidity in %
    Nsuns : numpy.array or None
        Number of suns intensity
    Encaps : numpy.array or None
        Encapsulation feature value in mg/ml PMMA
    O2 : numpy.array or None
        % of O2
    N2 : numpy.array or None
        % of N2
    use_default_minmax : bool
        If True, then the default values listed in the beginning
        of the file will be used for normalizing the values. Otherwise, the min and maximum
        of the input arguement arrays will be used
        
    Returns
    -----------------
    The list of dictionaries corresponding to the marker styles of each data point
    
    """
    
    cmap_edge = mpl.cm.get_cmap('jet')
    cmap_fill = mpl.cm.get_cmap('Greys')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    # Define which minimum and maximum values to use for scaling
    if use_default_minmax:
        minmax_T = TEMP
        minmax_RH = REL_HUM
        minmax_Nsuns = NSUNS
        minmax_encap = ENCAP
    else:
        minmax_T = None
        minmax_RH = None
        minmax_Nsuns = None
        minmax_encap = None
    
    
    # Encapsulation determined by fill color
    if not Encaps is None:
        Encaps_norm = tools.normalize(1-np.array(Encaps), pix_range=(0, 0.8), minmax=minmax_encap)
        c = norm([cmap_fill(i) for i in Encaps_norm])
    else:
        c = 'w'
    
    # edgecolor determinined by temperature
    if not T is None:
        T_norm = tools.normalize(T, minmax=minmax_T)
        edgecolor = norm([cmap_edge(i) for i in T_norm])
    else:
        edgecolor = 'k'
    
    # Nsuns determined by marker size
    if not Nsuns is None:
        Nsuns_norm = tools.normalize(np.log(1 + Nsuns), minmax=np.log(minmax_Nsuns+1))
        size = 100 + Nsuns_norm*350
    else:
        size = 100
    
    # edge_width and thus fill determined by relative humidity
    # change size and linewidth in such a way that the overall marker size stays constant
    if not RH is None:
        RH_norm = tools.normalize(RH, pix_range=(0.1, 0.6), minmax=minmax_RH)
        def props_for_RH(RH_norm):
            # returns linewidth to fully fill the marker, but size changes
            p1 = np.array([ 2.14415400e-08, -4.77467295e-05,  5.26042769e-02,  4.70106892e+00])
            linewidth_for_fullfill = lambda x : np.matmul(p1, [x**n for n in np.arange(len(p1)-1,-1,-1)])
            
            # returns size of marker for full fill when used with above function for linewidth
            p2 = np.array([-9.33333333e-08,  1.03736264e-04,  2.67263004e-01, -7.66153846e+00])
            same_size_fullfill = lambda x : np.matmul(p2, [x**n for n in np.arange(len(p2)-1,-1,-1)])
            
            curr_size = size + RH_norm*(same_size_fullfill(size)-size)
            line_width = RH_norm*(linewidth_for_fullfill(same_size_fullfill(size)))
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
            air_ind = np.logical_and(gas<80, gas>70)
            N2_ind = (gas == 100)
            O2_ind = (gas == 0)
            
        elif gas_type == 'O2':
            air_ind = np.logical_and(gas<30, gas>20)
            N2_ind = (gas == 0)
            O2_ind = (gas == 100)
        
        gas = gas.astype('<U1')
        gas[air_ind] = 'o' # air
        gas[N2_ind] = '^'  # N2
        gas[O2_ind] = 'v'  # O2
        gas_markers = gas
    else:
        gas_markers = ["o" for i in range(len(T))]
    
    dict_list = []
    for i in range(len(T)):
        dict_list += [{
                    'c' : c[i],
                    's': s[i],
                    'linewidths': linewidth[i],
                    'edgecolors': edgecolor[i],
                    'marker' : gas_markers[i]
        }]
        
    return dict_list
