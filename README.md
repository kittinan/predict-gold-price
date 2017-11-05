# Predict Gold Price
Simple gold price prediction with LSTM. We predict gold price up and down everyday at [https://kittinan.github.io/predict-gold-price](https://kittinan.github.io/predict-gold-price/)

### Install Requirement

```bash
pip install -r requirements.txt
```

### Run Jupyter

```bash
jupyter notebook
```
All code available in jupyter notebook [gold.ipynb](/gold.ipynb)

### Data
Scrape data price every day from [https://www.goldtraders.or.th](https://www.goldtraders.or.th/) in Thai Baht currency

![Thai Gold Price](/images/gold_price.png)

### Split data for test

![Split price data](/images/split.png)

### Predict 
Train for 10 epochs

![Predict](/images/predict.png)

## License
[MIT](/LICENSE)
