from neuralprophet import NeuralProphet
import pandas as pd

df = pd.DataFrame()

# manual split
m = NeuralProphet()
df_train, df_test = m.split_df(df, valid_p=0.2)

train_metrics = m.fit(df_train)
test_metrics = m.test(df_test)

# built-in function
m = NeuralProphet()
metrics = m.fit(df_train, validation_df=df_test)
