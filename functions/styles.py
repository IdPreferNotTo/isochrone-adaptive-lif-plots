from matplotlib import rc
from matplotlib import rcParams

colors = ["#173F5F", "#20639B", "#3CAEA3", "#F6D55C", "#BF6B63", "#D9A384"]

def color_palette():
    return ["#173F5F", "#20639B", "#3CAEA3", "#F6D55C", "#BF6B63", "#D9A384", "#ED553B"]




def figsize(s = 0.5):
    defaultwidth = 6.4
    defaultheight = 4.8
    textwidth = s*5.59164
    textheight = defaultheight*(textwidth/defaultwidth)
    return textwidth, textheight

def set_default_plot_style():
    rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Latin Modern'
    rcParams.update({'font.size': 11})
    rc('text', usetex=True)


def remove_top_right_axis(axis):
    for ax in axis:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

def remove_all_axis(axis):
    for ax in axis:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)