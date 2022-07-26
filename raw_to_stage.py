import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries as TS
from darts.models import VARIMA

from glob import glob

def read_file() -> pd.DataFrame:
	df = pd.DataFrame({'date': pd.date_range(start='01/01/2018', end='20/07/2022', )})

	# foreign price
	for fileName in glob('raw/*Daily.csv'):
		tmp_df = pd.read_csv(fileName, skiprows=4)
		tmp_df['Day'] = pd.to_datetime(tmp_df['Day'], format='%m/%d/%Y')
		tmp_df = tmp_df[tmp_df['Day'] > '01/01/2018']
		df = pd.merge(left=df, right=tmp_df, how='left', left_on='date', right_on='Day')
	df.drop(['Day_x', 'Day_y', 'Day'], axis=1, inplace=True)
	df.columns = list(map(lambda col: col.lower().replace('/', ':').replace(' ', '_'), list(df.columns)))
	
	# thai price
	thai_filename = 'raw/thai_oil_price.csv'
	thai_df = pd.read_csv(thai_filename)
	thai_df['date'] = pd.to_datetime(thai_df['date'], format='%d/%m/%Y')
	# select target price (gasohol95)
	select_cols = ['gasohol95']
	thai_df.drop(thai_df.columns.difference(['date'] + select_cols), axis=1, inplace=True)
	df = pd.merge(df, thai_df, how='left', on='date')

	# USD/THB for reference
	exchange_rate_df = pd.read_csv('raw/HistoricalPrices.csv', delimiter=', ', engine='python')
	exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'], format='%m/%d/%y')
	df = pd.merge(df, exchange_rate_df, how='left', left_on='date', right_on='Date')
	df.drop(['Date', 'Open', 'High', 'Low'], axis=1, inplace=True)
	df.rename(columns={'Close': 'usd_thb'}, inplace=True)

	# manipulate input_data (add price change column and forward fill)
	to_adjust_cols = ['wti_cushing_oklahoma:crude_oil:dollars_per_barrel', 'brent_europe:crude_oil:dollars_per_barrel', 'us_gulf_coast:conventional_gasoline:dollars_per_gallon']
	for col in to_adjust_cols:
		prefix_col = col.split(':')[0].split('_')[0]
		df[f'{prefix_col}_price_change'] = df[col].notna()
		df[col].ffill(inplace=True)

	df['target_price_change'] = df['gasohol95'].notna()
	df['gasohol95'].ffill(inplace=True)

	# filter for only rows that contain values
	df = df[df['date'] > '01/10/2018']
	
	# print(df.head(20))
	# print(df.tail(20))
	# print(df.describe())
	# print(df.dtypes)
	# df.to_csv('dump.csv', index=False)
	return df

def forecast(df: pd.DataFrame):
	plt.figure(figsize=(19.2,10.8))

	series = TS.from_dataframe(df, time_col='date')
	series.plot()
	# pick_last_n_date = 15
	# train, test = series[:-pick_last_n_date], series[-pick_last_n_date:]
	# model = VARIMA()
	# model.fit(train)
	# prediction = model.predict(len(test))
	# prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
	plt.legend()
	plt.savefig('plot.png')

def main():
	df = read_file()
	# print(df)
	forecast(df)


if __name__ == '__main__':
	main()
