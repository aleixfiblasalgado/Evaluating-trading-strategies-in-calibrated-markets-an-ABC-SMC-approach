rm cache/*\.\.log*.pickle
rm -rf visualizations
python asset_returns_stylized_facts.py -r ../data/1m_ohlc/ohlc_AAPL_2022-06-26.bz2 -s ../log/random_fund_value -s ../log/random_fund_diverse -s ../log/hist_fund_value -s ../log/hist_fund_diverse