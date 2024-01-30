import os
import requests
import gzip
import shutil
import tarfile

def download_mnist():
    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ]

    if not os.path.exists('data'):
        os.makedirs('data')

    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            print(f"Downloading {url}...")
            r = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                f.write(r.content)
        print(f"Decompressing {filepath}...")
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def download_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if not os.path.exists('data'):
        os.makedirs('data')

    filename = url.split("/")[-1]
    filepath = os.path.join('data', filename)
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        r = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            f.write(r.content)
    print(f"Decompressing {filepath}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path='data')

if __name__ == "__main__":
    # download_mnist()
    download_cifar10()
