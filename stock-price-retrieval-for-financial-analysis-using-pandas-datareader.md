# Stock Price retrieval for Financial Analysis using Pandas-datareader

The stock price of Tesla can be retrieved for further analysis along with the sentiment analysis.

Yahoo provides stock data and the following endpoints available are as follows:

* `yahoo` - retrieve daily stock prices (high, open, close, volume and adjusted close)
* `yahoo-actions` - retrieve historical corporate actions (dividends and stock splits)
* `yahoo-dividends` - retrieve historical dividends

### Retrieve daily stock price

Import relavent modules

```
import matplotlib
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib as mpl
import matplotlib.pyplot as plt
```

DataReader takes four essential arguments

* `name` - names of target stock price (a list).
* `data_source` - target data source eg. yahoo finance endpoints `"yahoo", "yahoo-actions", "yahoo-dividends"`
* `start` - left boundary for retrieval range in (defaults to 1/1/2010)
* `end` - right boundary for retrieval range (defaults to today)

<mark style="background-color:orange;">start</mark> and <mark style="background-color:orange;">end</mark> can be string, int, date, datetime, Timestamp.

![Arguments of DataReader function source: https://github.com/pydata/pandas-datareader/blob/ab26ad2099d6a7bc79447e71d72cdb93d8299b3c/pandas\_datareader/data.py#L274](<.gitbook/assets/image (2).png>)

```
test = data.DataReader(['TSLA'], 'yahoo', start='2021/01/01', end='2021/12/31')
```

### Output

The output of retrieved stock prices.

![Daily stock prices of Tesla](<.gitbook/assets/image (15).png>)

Save it to CSV file for further analysis

```
test.to_csv('tesla_2021_prices.csv')
```

Bibliography

cjhutto 2019. GitHub - cjhutto/vaderSentiment: VADER Sentiment Analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains. Available at: https://github.com/cjhutto/vaderSentiment \[Accessed: 31 March 2022].

Hutto, C. and Gilbert, E. \[no date]. View of VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Available at: https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399 \[Accessed: 31 March 2022].

pushshift \[no date]. GitHub - pushshift/api: Pushshift API. Available at: https://github.com/pushshift/api \[Accessed: 31 March 2022].

pydata \[no date]. pandas-datareader/data.py at ab26ad2099d6a7bc79447e71d72cdb93d8299b3c · pydata/pandas-datareader. Available at: https://github.com/pydata/pandas-datareader/blob/ab26ad2099d6a7bc79447e71d72cdb93d8299b3c/pandas\_datareader/data.py#L274 \[Accessed: 1 April 2022].

Remote Data Access — pandas-datareader 0.10.0 documentation. \[no date]\[a]. Available at: https://pandas-datareader.readthedocs.io/en/latest/remote\_data.html \[Accessed: 31 March 2022].

Remote Data Access — pandas-datareader 0.10.0 documentation. \[no date]\[b]. Available at: https://pandas-datareader.readthedocs.io/en/latest/remote\_data.html#remote-data-yahoo \[Accessed: 1 April 2022].

Tausczik, Y.R. and Pennebaker, J.W. 2009. The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods. _Journal of Language and Social Psychology_ 29(1), pp. 24–54. doi: 10.1177/0261927x09351676.
