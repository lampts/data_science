Tricks for productivity
----

- class_weight for imbalance

``` python
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}
```

- train skipthoughts

sudo THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python st_finser.py


- unzip glove word2vec

sudo port install p7zip
brew install p7zip
7za x file.zip

- iterm 2 + omzsh: https://gist.github.com/kevin-smets/8568070

- numpy array as pandas column: https://stackoverflow.com/questions/18646076/add-numpy-array-as-column-to-pandas-data-frame
