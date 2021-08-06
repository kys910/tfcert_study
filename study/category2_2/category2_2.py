import tensorflow_datasets as tfds
import requests

def downloadDataOnline(url):
    r=requests.get(url, allow_redirects=True)
    open('iris.data', 'wb').write(r.content)
downloadDataOnline('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

df=pd.read_csv('iris.data',header=None,names=['F1','F2','F3','F4','Label'])
df.head()


'''
train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[-20%:]')
'''