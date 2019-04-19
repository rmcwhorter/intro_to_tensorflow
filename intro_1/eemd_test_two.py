from PyEMD import EEMD, CEEMDAN
import numpy as np
import pandas as pd 
from bokeh.plotting import figure, output_file, show

df = pd.read_csv(f'training_datas/ETH-USD.csv', names=['time', 'low', 'high', 'open', 'close', 'volume'])

# output to static HTML file
output_file("viz/lines.html")

eemd = EEMD()
trials = 100
eemd.trials = trials
eemd.noise_width = 0.08     #np.std(df['close'].values[300:300+(60*5)])
print(eemd.noise_width)

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label=f"{trials}", y_axis_label='price', plot_width=1000, plot_height=1000)





'''
for a in df.columns.values[1:-1]:
    eemd_arrays = eemd(df[a].values[300:300+(60*5)])
    p.line(range(len(eemd_arrays[-1])), df[a].values[300:300+(60*5)], legend=f"{a}", line_width=2, line_color=colors[count % len(colors)])
    p.line(range(len(eemd_arrays[-1])), eemd_arrays[-1] + df[a].iloc[300] - eemd_arrays[-1][0], legend=f"EEMD-{a}", line_width=2, line_color=colors[count % len(colors)])
    count += 1
'''
#series = eemd(df['close'].values[300:300+(60*5)])
colors = ['blue', 'red', 'yellow', 'green', 'purple', 'orange', 'black']
count = 0
for a in [0.01, 0.025, 0.05, 0.075, 0.1, 0.11, 0.12]:
    eemd.noise_width = a
    series = eemd(df['close'].values[300:300+(60*5)])
    p.line(range(len(series[-1])), series[-1] + df['close'].values[300] - series[-1][0], legend=f"EEMD-{a}", line_color = colors[count % len(colors)])
    count += 1

p.line(range(len(df['close'].values[300:300+(60*5)])), df['close'].values[300:300+(60*5)])



show(p)




