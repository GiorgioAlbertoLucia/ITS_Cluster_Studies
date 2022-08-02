import pandas as pd
from ROOT import TH2F, TCanvas, TH1F, gStyle, kBird, gPad, gROOT, TLegend

def set_obj_style(obj, **kwargs):

    if 'set_opt_stat' in kwargs:                gStyle.SetOptStat(kwargs['set_opt_stat'])
    else:                                       gStyle.SetOptStat(1110000)

    if 'title' in kwargs:                       obj.SetTitle(kwargs['title'])
    if 'x_label' in kwargs:                     obj.GetXaxis().SetTitle(kwargs['x_label'])
    if 'y_label' in kwargs:                     obj.GetYaxis().SetTitle(kwargs['y_label'])
    

    if 'linecolor' in kwargs:                   obj.SetLineColor(kwargs['linecolor'])
    else:                                       obj.SetLineColor(1)

    if 'top_margin' in kwargs:                  gPad.SetTopMargin(kwargs['top_margin'])
    else:                                       gPad.SetTopMargin(0.15)

    if 'left_margin' in kwargs:                 gPad.SetLeftMargin(kwargs['left_margin'])
    else:                                       gPad.SetLeftMargin(0.15)

    if 'right_margin' in kwargs:                gPad.SetRightMargin(kwargs['right_margin'])
    else:                                       gPad.SetRightMargin(0.15)




def fill_hist(hist, x, y=None):

    if type(y) == pd.Series and y.empty:
        for element in x:                           hist.Fill(element)
    elif type(y) != pd.Series and y == None:    
        for element in x:                           hist.Fill(element)
    else:
        for (element_x, element_y) in zip(x, y):    hist.Fill(element_x, element_y)