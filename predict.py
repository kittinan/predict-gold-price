import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import re
import urllib.request
import os
import jinja2
import sys

import keras
from keras.models import load_model

def predict_torrmow():
    model = load_model('model.h5')

    df = pd.read_csv('golds.csv')
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    train = df[["price"]]
    sc = MinMaxScaler()

    train_sc = sc.fit_transform(train)
    X_train = train_sc[:-1]
    y_train = train_sc[1:]
    X_train_t = X_train[:, None]

    y_pred = model.predict(X_train_t)
    result = sc.inverse_transform(y_pred)[-2:]
    today = round(result[0][0])
    tomorrow = round(result[1][0])
    output = "flat"
    if tomorrow > today:
        output = "up"
    elif tomorrow < today:
        output = "down"
    return output, tomorrow

def load_gold_price():

    url = "https://www.goldtraders.or.th/default.aspx"
    html = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla'})).read().decode("utf-8")

    date = re.findall('<span id="DetailPlace_uc_goldprices1_lblAsTime".*?>(\d\d/\d\d/25\d\d).*?</span>', html)
    date = date[0] if len(date) else None
    day, month, thai_year= date.split("/")
    year = int(thai_year) - 543


    price = re.findall('<span id="DetailPlace_uc_goldprices1_lblBLSell".*?>(.*?)</span>', html)
    price = price[0] if len(price) else None
    price = float(price.replace(",", ""))

    return "{}-{}-{}".format(year, month, day), price

def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)

def render_predict(status, tomorrow):
    df_predict = pd.read_csv("predicted.csv")
    predicts = df_predict.sort_index(ascending=False).to_dict('records')
    data = {"status": status, "tomorrow": tomorrow, "predicts": predicts}
    html = render('./docs/template.html', data)

    with open("./docs/index.html", "w") as html_file:
        html_file.write(html)

df = pd.read_csv('golds.csv')

df.columns = ['date', 'price']
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

gold_date, gold_price = load_gold_price()

if gold_date is None or gold_price is None:
    print("Can not get gold price.")
    sys.exit()

print("{}: {}".format(gold_date, gold_price))

if gold_date in df.index:
    print("Exist date: {}".format(gold_date))
    print("Exit.")
    sys.exit()

# Save new data
new_df_gold = pd.DataFrame([[gold_date, gold_price]], columns=["date","price"])
new_df_gold['date'] = pd.to_datetime(new_df_gold['date'])
new_df_gold = new_df_gold.set_index('date')

df = df.append(new_df_gold)
df["price"] = df["price"].astype(float)
df.to_csv("golds.csv")

sign, price = predict_torrmow()

x = pd.DatetimeIndex(df.iloc[-1:].index.values) + pd.DateOffset(1)
ts = pd.to_datetime(x[0])
tomorrow = ts.strftime('%Y-%m-%d')

new_df_predict = pd.DataFrame([[tomorrow, sign, round(price, 2), -1]], columns=["date","trend","price","is_correct"])

df_predict = pd.read_csv("predicted.csv")

# Check Predict
check_date = pd.DatetimeIndex(df_predict.iloc[-1:]["date"].values).strftime('%Y-%m-%d')[0]
try:
    days = df.iloc[df.index.get_loc(check_date)-1: df.index.get_loc(check_date)+1]
except:
    days = df.iloc[-2:]
today_price = days.iloc[1]["price"]
yesterday_price = days.iloc[0]["price"]
real_sign = "flat"
if today_price > yesterday_price:
    real_sign = "up"
elif today_price < yesterday_price:
    real_sign = "down"

is_correct = df_predict[df_predict["date"] == check_date]["trend"].values[0] == real_sign

# Set value
df_predict.set_value(df_predict[df_predict["date"] == check_date].index[0], "is_correct", is_correct)
print("check_date: {} = {}".format(check_date, is_correct))

df_predict = df_predict.append(new_df_predict, ignore_index=True)
df_predict.to_csv("predicted.csv", index=False)

predicts = df_predict.sort_index(ascending=False).T.to_dict().values()

render_predict(sign, tomorrow)

print("{}: {}".format(sign, price))
