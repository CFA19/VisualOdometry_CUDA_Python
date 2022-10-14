from cProfile import label
from pickle import TRUE
from matplotlib.axis import Axis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
from bokeh.models import Div, WheelZoomTool
from bokeh.models.widgets import Panel, Tabs
from sklearn.metrics import mean_squared_error
from math import sqrt
from sqlalchemy import false
from torch import gt

def visualize_paths(gt_path, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"

   

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)

    #RMSE = np.sqrt(mean_squared_error(gt_path, pred_path))
    RMSE =np.sqrt(mean_squared_error(gt_path, pred_path))
    print("El valor del RMSE es :", RMSE)
    print("El valor 2DRMS_full es :", 2*RMSE)

    summation = 0
    n = len(gt_x) #finding total number of items in list
    for i in range (0,n):
        #looping through each element of the list
        #difference = (gt_x[i]**2) + (gt_y[i]**2)  #finding the difference between observed and predicted value
        #difference2 = (gt_path[i]**2) + (pred_path[i]**2)
        difference = (gt_x[i]-pred_x[i])**2 + (gt_y[i]-pred_y[i])**2
        #squared_difference = difference**2  #taking square of the differene 
        summation = summation + difference  #taking a sum of all the differences
    MSE2=summation/(n*2)
    RMSE2=sqrt(MSE2)
    print ("The Root Mean Square Error CODE is: ", RMSE2)

    source = ColumnDataSource(data=dict(gtx=gt_path[:, 0], gty=gt_path[:, 1],
                                        px=pred_path[:, 0], py=pred_path[:, 1],
                                        diffx=np.arange(len(diff)), diffy=diff,
                                        disx=xs, disy=ys))

    
    fig1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="left",
                  x_axis_label="x", y_axis_label="y")

    
       
    fig1.axis.major_label_text_font_size = '40px'
    fig1.axis.axis_label_text_font_style = 'bold'
    #fig1.legend.title_text_font_size = "28px"
    
    fig1.circle("px", "py", source=source, color="blue", hover_fill_color="firebrick", legend_label="Prediction")
    fig1.line("px", "py", source=source, color="blue", legend_label="Prediction",  line_width = 4)
    fig1.legend.location = "top_left"
    fig1.legend.title_text_font = 'Times New Roman'
    fig1.legend.label_text_font_size = '28pt'
    #fig1.legend.label_text_font_style = 'bold'
    #fig1.legend.title_text_font_style = "bold"
    #fig1.legend.title_text_font_size = "28pt"
    #fig1.legend.title_text_font = 'Arial'
    #fig1.legend.title_text_font_size = '60pt'

    fig1.square("gtx", "gty", source=source, color="black", hover_fill_color="firebrick", legend_label="Ground-Truth")
    fig1.line("gtx", "gty", source=source, color="black", legend_label="Ground-Truth", line_width = 4)
    fig1.legend.location = "top_left"
    #fig1.legend.title_text_font_style = "bold"
    #fig1.legend.title_text_font_size = "28px"
    fig1.legend.title_text_font = 'Times New Roman'
    fig1.legend.title_text_font_size = '28pt'
   # fig1.legend.title_text_font_style = 'bold'
    

    fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed",  line_width = 1)
    fig1.legend.click_policy = "hide"
    fig1.legend.location = "top_left"
    fig1.legend.title_text_font = 'Times New Roman'
    fig1.legend.title_text_font_size = '28pt'
  #  fig1.legend.title_text_font_style = 'bold'
    

    fig2 = figure(title="Error", tools=tools, width_policy="max", toolbar_location="left",
                  x_axis_label="frame", y_axis_label="error")
    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h1>Paths</h1>"),
                 [fig1, fig2],
                 ], sizing_mode='scale_width'))
    

def make_residual_plot(x, residual_init, residual_minimized):
    fig1 = figure(title="Initial residuals", x_range=[0, len(residual_init)], x_axis_label="residual", y_axis_label="")
    fig1.line(x, residual_init)

    change = np.abs(residual_minimized) - np.abs(residual_init)
    plot_data = ColumnDataSource(data={"x": x, "residual": residual_minimized, "change": change})
    tooltips = [
        ("change", "@change"),
    ]
    fig2 = figure(title="Optimized residuals", x_axis_label=fig1.xaxis.axis_label, y_axis_label=fig1.yaxis.axis_label,
                  x_range=fig1.x_range, y_range=fig1.y_range, tooltips=tooltips)
    fig2.line("x", "residual", source=plot_data)

    fig3 = figure(title="Change", x_axis_label=fig1.xaxis.axis_label, y_axis_label=fig1.yaxis.axis_label,
                  x_range=fig1.x_range, tooltips=tooltips)
    fig3.line("x", "change", source=plot_data)
    return fig1, fig2, fig3


def plot_residual_results(qs_small, small_residual_init, small_residual_minimized,
                          qs, residual_init, residual_minimized):
    output_file("plot.html", title="Bundle Adjustment")
    x = np.arange(2 * qs_small.shape[0])
    fig1, fig2, fig3 = make_residual_plot(x, small_residual_init, small_residual_minimized)

    x = np.arange(2 * qs.shape[0])
    fig4, fig5, fig6 = make_residual_plot(x, residual_init, residual_minimized)

    show(layout([Div(text="<h1>Bundle Adjustment exercises</h1>"),
                 Div(text="<h1>Bundle adjustment with reduced parameters</h1>"),
                 gridplot([[fig1, fig2, fig3]], toolbar_location='left'),
                 Div(text="<h1>Bundle adjustment with all parameters (with sparsity)</h1>"),
                 gridplot([[fig4, fig5, fig6]], toolbar_location='left')
                 ]))


def plot_sparsity(sparse_mat):
    fig, ax = plt.subplots(figsize=[60, 40])
    plt.title("Sparsity matrix")

    ax.spy(sparse_mat, aspect="auto", markersize=0.02)
    plt.xlabel("Parameters")
    plt.ylabel("Resudals")

    plt.show()
