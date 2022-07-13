import requests
from shutil import unpack_archive

def download_data():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

    with open('bank-additional.zip', 'wb') as f:
        f.write(requests.get(data_url).content)

    unpack_archive('bank-additional.zip')
