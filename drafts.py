import plotly.graph_objects as go  # for data visualisation
import plotly
from plotly.subplots import make_subplots
import plotly.io as pio


def paramSpace(df):

    zmax=df[["Delta", "Theta", "Alpha", "Beta", "Gamma"]].max().max()
    zmin=df[["Delta", "Theta", "Alpha", "Beta", "Gamma"]].min().min()

    fig = make_subplots(rows=1, cols=5, subplot_titles=( "Delta", "Theta", "Alpha", "Beta", "Gamma"),
                        specs=[[{},  {},  {},  {},  {}]], shared_yaxes=True, shared_xaxes=True)

    fig.add_trace(go.Heatmap(z=df.plvD_r,x=df.speed,y=df.G,colorscale='RdBu',reversescale=True, zmin=zmin,zmax=zmax), row=1, col=1)
    fig.add_trace(go.Heatmap(z=df.plvT_r,x=df.speed,y=df.G,colorscale='RdBu',reversescale=True,zmin=zmin,zmax=zmax), row=1, col=2)
    fig.add_trace(go.Heatmap(z=df.plvA_r,x=df.speed,y=df.G,colorscale='RdBu',reversescale=True,zmin=zmin,zmax=zmax), row=1, col=3)
    fig.add_trace(go.Heatmap(z=df.plvB_r,x=df.speed,y=df.G,colorscale='RdBu',reversescale=True, zmin=zmin,zmax=zmax),row=1, col=4)
    fig.add_trace(go.Heatmap(z=df.plvG_r,x=df.speed,y=df.G,colorscale='RdBu',reversescale=True,zmin=zmin,zmax=zmax), row=1, col=5)

    fig.update_layout(title_text='FC correlation (emp-sim) by speed and coupling factor')
    pio.write_html(fig, file="figures/paramSpace.html", auto_open=True)


number_list = [1, 2, 3,5]
str_list = ['one', 'two', 'three',"zwro"]
a=[21,22,23,0,8]
result = zip(number_list, str_list,a)
l=list(result)
l