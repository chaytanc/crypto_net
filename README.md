# crypto_net

This repository has a feed forward neural network and a whole lot of data processing on cryptocurrency stock prices. It uses Pytorch. It first begins with unsupervised kmeans cluster based on the volumes' volatility/std dev per company. Many of these settings are adjustable in parameters.py
You may want to set the param 'load_processing' to False if this is the first time you're running the program.
