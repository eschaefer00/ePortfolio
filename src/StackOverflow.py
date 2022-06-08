from neuralprophet import NeuralProphet
import pandas as pd
from neuralprophet.df_utils import check_dataframe

# data import
df = pd.read_csv(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\data\stackOverflow.csv', sep=',')

# PYTHON

# drop columns that are not needed for analysis
df_py = df.drop(columns=['machine-learning', 'pandas'])
# rename columns to fit the NeuralProphet conditions
df_py.columns = ['ds', 'y']

# instantiate NeuralProphet model
m_py = NeuralProphet()
# fit the model
metrics_py = m_py.fit(df=df_py, freq='MS')

# create future dataframe based on df
fut_py = m_py.make_future_dataframe(df=df_py, periods=36)
fut_py.columns = ['ds', 'y']

new_py = pd.concat([df_py, fut_py])

# check dataframe and make it ready for prediction
chckd_py = check_dataframe(new_py, check_y=True, covariates=None, regressors=None, events=None)
# make prediction on entire data
future_forecast_py = m_py.predict(chckd_py)

# plotting
fig_fcst_py = m_py.plot(future_forecast_py, ax=None, xlabel='time', ylabel='comments', figsize=(10, 6))
fig_comp_py = m_py.plot_components(future_forecast_py)
fig_param_py = m_py.plot_parameters(weekly_start=0, yearly_start=0, figsize=None, df_name=None)

# save figures
# fig_fcst_py.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\fig_fcst_py.png')
# fig_comp_py.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\fig_comp_py.png')
# fig_param_py.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\fig_param_py.png')

# MACHINE LEARNING

# drop columns that are not needed for analysis
df_ml = df.drop(columns=['python', 'pandas'])
# rename columns to fit the NeuralProphet conditions
df_ml.columns = ['ds', 'y']

# instantiate NeuralProphet model
m_ml = NeuralProphet()
# fit the model
metrics_ml = m_ml.fit(df=df_ml, freq='MS')

# create future dataframe based on df
fut_ml = m_ml.make_future_dataframe(df=df_ml, periods=36)
fut_ml.columns = ['ds', 'y']
# concat dfs
new_ml = pd.concat([df_ml, fut_ml])
# check dataframe and make it ready for prediction
chd_ml = check_dataframe(new_ml, check_y=True, covariates=None, regressors=None, events=None)
# make prediction on entire data set
future_forecast_ml = m_ml.predict(chd_ml)

# plotting
fig_fcst_ml = m_ml.plot(future_forecast_ml, ax=None, xlabel='time', ylabel='comments', figsize=(10, 6))
fig_comp_ml = m_ml.plot_components(future_forecast_ml)
fig_param_ml = m_ml.plot_parameters(weekly_start=0, yearly_start=0, figsize=None, df_name=None)

# save figures
# fig_fcst_ml.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\forecast_ml.png')
# fig_comp_ml.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\components_ml.png')
# fig_param_ml.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\parameters_ml.png')

# PANDAS

# drop columns that are not needed for analysis
df_pd = df.drop(columns=['python', 'machine-learning'])
# rename columns to fit the NeuralProphet conditions
df_pd.columns = ['ds', 'y']

# instantiate NeuralProphet model
m_pd = NeuralProphet()
metrics_pd = m_pd.fit(df=df_pd, freq='MS')

# create future dataframe based on df
fut_pd = m_pd.make_future_dataframe(df=df_pd, periods=36)
fut_pd.columns = ['ds', 'y']
# concat past and future dfs
new_pd = pd.concat([df_pd, fut_pd])

# check dataframe and make it ready for prediction
chd_pd = check_dataframe(new_pd, check_y=True, covariates=None, regressors=None, events=None)
# prediction on entire dataframe
future_forecast_pd = m_pd.predict(chd_pd)

# plotting
fig_fcst_pd = m_pd.plot(future_forecast_pd, ax=None, xlabel='time', ylabel='comments', figsize=(10, 6))
fig_comp_pd = m_pd.plot_components(future_forecast_pd)
fig_param_pd = m_pd.plot_parameters(weekly_start=0, yearly_start=0, figsize=None, df_name=None)

# save figures
# fig_fcst_pd.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\forecast_pd.png')
# fig_comp_pd.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\components_pd.png')
# fig_param_pd.savefig(r'C:\Users\Elisa Schäfer\Documents\DHBW\Code\SoftwareEngineering\ePortfolio\results\parameters_pd.png')
