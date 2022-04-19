# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + language="javascript"
# var rate = 0;
# // apply setting to  all current CodeMirror instances
# IPython.notebook.get_cells().map(
#     function(c) {  return c.code_mirror.options.cursorBlinkRate=rate;  }
# );
# // make sure new CodeMirror instance also use this setting
# CodeMirror.defaults.cursorBlinkRate=rate;
# -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_excel('./data/Статистические_данные_показателей_СЭР.xlsx', index_col=0).T

df.columns.values[72] += ' 1'
df.columns.values[73] += ' 1'

df.iloc[:,69:80]

df_no_na = df.dropna(axis=1, thresh=12)

df_no_na.isnull().sum(axis = 0).sort_values(ascending=False)

df.isnull().sum(axis = 0).value_counts().sort_index(ascending=False)

df_no_idx = df_no_na[df_no_na.columns.drop(list(df_no_na.filter(regex='Индекс.*')))]
df_renamed = df_no_idx.rename({col: ' '.join(el[:5].capitalize() if len(el)>1 else el for el in col.split(' '))[:60] for col in df_no_idx.columns}, axis=1)

for i in range(int(df_renamed.shape[1]/10)+1):
    display(df_renamed.iloc[:,i*10:(i+1)*10])

for i in range(df_renamed.shape[1]):
    _, ax = plt.subplots()
    ax.plot(df_renamed.iloc[:,i])
    ax.set_title(df_renamed.iloc[:,i].name)

# +
plt.style.use('seaborn')
sns.set(font_scale=1)

pp = PdfPages('visualization.pdf')
plt.style.use('seaborn')
sns.set(font_scale = 0.8)
sns.set_style('whitegrid')

for i in range(int(df_renamed.shape[1]/4)+1):
    num = i*4
    plt.figure(figsize=(11.69,8.27),dpi=200)
    for j in range(min(4, df_renamed.shape[1]-num)):
        plt.subplot(2, 2, 1+j)
        plt.xticks(np.arange(int(df_renamed.index[0]/5)*5, int(df_renamed.index[-1]/5)*5+5, 5))
        sns.lineplot(x=df_renamed.index, y=df_renamed.iloc[:, num+j].name, data=df_renamed,  ci=None, color='g', marker='o')
    pp.savefig()
pp.close()
# -

df_demographic = df_renamed.iloc[:,:8].drop(df_renamed.columns[[1,6]], axis=1)
df_economic = df_renamed.iloc[:,8:18].drop(df_renamed.columns[[10,15]], axis=1) # Trended
df_trade = df_renamed.iloc[:,18:24]
df_finance = df_renamed.iloc[:,24:32].drop(df_renamed.columns[25], axis=1) # Trended

df_demographic.head(3)

df_economic.tail(3)

df_trade.tail(3)

df_finance.tail(3)


