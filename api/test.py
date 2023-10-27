import pycaret.regression as rgr
import pandas as pd
df=pd.read_csv("test_data\housing.csv")

rgr.setup(df, target=chosen_target, silent=True)
setup_df = rgr.pull()
print(set)
best_model = rgr.compare_models()
compare_df = rgr.pull()
rgr.save_model(best_model, 'best_model')