import os
import sys
import gzip
import shutil
from six.moves import urllib



# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

def download_progress(count,blocksize,totalsize):
    sys.stdout.write("\r>> Downloading %.1f%%" % ( float(count * blocksize) / float(totalsize) * 100.))
    sys.stdout.flush()

def download_and_uncompress(URL,saveto,force):
    '''
       Args:
           URL: the download links for data
           dataset_dir: the path to save data
           force: re-download data
       '''
    filename = URL.split('/')[-1]
    filepath = os.path.join(saveto, filename)
    if not os.path.exists(saveto):
        os.mkdir(saveto)
    extract_to = os.path.splitext(filepath)[0]

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
        print('Extracting ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('Successfully extracted')
        print()


def start_download(dataset,saveto,forceflag):
    if dataset == 'mnist':
        pathname = os.path.join(saveto,'mnist')
        download_and_uncompress(MNIST_TRAIN_IMGS_URL,pathname,forceflag)
        download_and_uncompress(MNIST_TRAIN_LABELS_URL,pathname,forceflag)
        download_and_uncompress(MNIST_TEST_IMGS_URL,pathname,forceflag)
        download_and_uncompress(MNIST_TEST_LABELS_URL,pathname,forceflag)
    elif dataset == 'fashion_mnist':
        pathname = os.path.join(saveto, 'fashion_mnist')
        download_and_uncompress(FASHION_MNIST_TRAIN_IMGS_URL,pathname,forceflag)
        download_and_uncompress(FASHION_MNIST_TRAIN_LABELS_URL,pathname,forceflag)
        download_and_uncompress(FASHION_MNIST_TEST_IMGS_URL,pathname,forceflag)
        download_and_uncompress(FASHION_MNIST_TEST_LABELS_URL,pathname,forceflag)
    else:
        raise Exception("Invalid dataset name! please check it: ", dataset)

'''独立运行'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for downloading datasets.')
    parser.add_argument("--dataset",default='mnist',choices=['mnist','fashion_mnist'],help='待下载的数据集名称')
    parser.add_argument("--saveto",default='data',help='下载后存放地址')
    parser.add_argument("--force",default=False,type=bool,help='强制覆盖本地已有文件')
    args=parser.parse_args()

    start_download(args.dataset,args.saveto,args.force)

    #with request.urlopen('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz') as f:
