from neuralprophet import NeuralProphet
import pandas as pd
from neuralprophet.df_utils import check_dataframe
import matplotlib.pyplot as plt

# data import
# read csv into DataFrame
df = pd.read_csv(r'/Users/elifant/PycharmProjects/worldsHappiness/data/stackOverflow.csv', sep=',')

# PYTHON

# drop columns that are not needed for analysis
df_py = df.drop(columns=['machine-learning', 'pandas'])
# rename columns to fit the NeuralProphet conditions
df_py.columns = ['ds', 'y']

# instantiate NeuralProphet model
m_py = NeuralProphet()
metrics_py = m_py.fit(df=df_py, freq='MS')

# fcst_py = m_py.predict(df_py)
# fig_py = m.plot(fcst_py)
# fig_py.savefig(r'/Users/elifant/PycharmProjects/worldsHappiness/res/fcst_py.png')

# create future dataframe based on df
fut_py = m_py.make_future_dataframe(df=df_py, periods=36)
fut_py.columns = ['ds', 'y']

new_py = pd.concat([df_py, fut_py])
pd.to_datetime(new_py['ds'])

# new.to_csv(r'/Users/elifant/PycharmProjects/worldsHappiness/data/stackOverflowNew.csv')
chd_py = check_dataframe(new_py, check_y=True, covariates=None, regressors=None, events=None)
future_forecast_py = m_py.predict(chd_py)

# plotting
fig_fcst_py = m_py.plot(future_forecast_py, ax=None, xlabel='time', ylabel='comments', figsize=(10, 6))
# fig_comp_py = m_py.plot_components(future_forecast)
# fig_param_py = m_py.plot_parameters(weekly_start=0, yearly_start=0, figsize=None, df_name=None)

# save figures
# fig_fcst_py.savefig(r'/Users/elifant/PycharmProjects/worldsHappiness/res/forecast_python.png')

# MACHINE LEARNING

# drop columns that are not needed for analysis
df_ml = df.drop(columns=['python', 'pandas'])
# rename columns to fit the NeuralProphet conditions
df_ml.columns = ['ds', 'y']

# instantiate NeuralProphet model
m_ml = NeuralProphet()
metrics_ml = m_ml.fit(df=df_ml, freq='MS')

# forecast = m_ml.predict(df_ml)
# fig_py = m.plot(forecast)
# fig_py.savefig(r'/Users/elifant/PycharmProjects/worldsHappiness/res/forecast2.png')

# create future dataframe based on df
fut_ml = m_ml.make_future_dataframe(df=df_ml, periods=36)
fut_ml.columns = ['ds', 'y']

new_ml = pd.concat([df_ml, fut_ml])
pd.to_datetime(new_ml['ds'])

chd_ml = check_dataframe(new_ml, check_y=True, covariates=None, regressors=None, events=None)
future_forecast_ml = m_ml.predict(chd_ml)

# plotting
fig_fcst_ml = m_ml.plot(future_forecast_ml, ax=None, xlabel='time', ylabel='comments', figsize=(10, 6))
# fig_comp_ml = m_ml.plot_components(future_forecast)
# fig_param_ml = m_ml.plot_parameters(weekly_start=0, yearly_start=0, figsize=None, df_name=None)

# save figures
# fig_fcst_ml.savefig(r'/Users/elifant/PycharmProjects/worldsHappiness/res/forecast_ml.png')

# PANDAS

# drop columns that are not needed for analysis
df_pd = df.drop(columns=['python', 'machine-learning'])
# rename columns to fit the NeuralProphet conditions
df_pd.columns = ['ds', 'y']

# instantiate NeuralProphet model
m_pd = NeuralProphet()
metrics_pd = m_pd.fit(df=df_pd, freq='MS')

# forecast = m_pd.predict(df_pd)
# fig_py = m.plot(forecast)
# fig_py.savefig(r'/Users/elifant/PycharmProjects/worldsHappiness/res/forecast2.png')

# create future dataframe based on df
fut_pd = m_pd.make_future_dataframe(df=df_pd, periods=36)
fut_pd.columns = ['ds', 'y']

new_pd = pd.concat([df_pd, fut_pd])
pd.to_datetime(new_pd['ds'])

# new.to_csv(r'/Users/elifant/PycharmProjects/worldsHappiness/data/stackOverflowNew.csv')
chd_pd = check_dataframe(new_pd, check_y=True, covariates=None, regressors=None, events=None)
future_forecast_pd = m_pd.predict(chd_pd)

# plotting
fig_fcst_pd = m_pd.plot(future_forecast_pd, ax=None, xlabel='time', ylabel='comments', figsize=(10, 6))
# fig_comp_pd = m_pd.plot_components(future_forecast)
# fig_param_pd = m_pd.plot_parameters(weekly_start=0, yearly_start=0, figsize=None, df_name=None)

# save figures
# fig_fcst_pd.savefig(r'/Users/elifant/PycharmProjects/worldsHappiness/res/forecast_pandas.png')

