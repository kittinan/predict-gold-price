#!/bin/bash

python3 predict.py
git add golds.csv predicted.csv docs/index.html
git commit -m "Update prediction"
git push origin master
