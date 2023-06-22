import os
from time import time as stdtime

import numpy as np
import sklearn
import joblib
from sklearn.cluster import MiniBatchKMeans as KMeans

class MiniBatchKMeans:
    '''
        Customized MiniBatch KMeans
    '''
    def __init__(self, n_clusters, init='k-means++', max_iter=150,
                batch_size=10000, tol=0.0, max_no_improvement=100,
                n_init=20, reassignment_ratio=0.5, random_state=None,
                pretrained_path=None, **kwargs):
        if pretrained_path is not None:
            print(f'Load pretrained model from {pretrained_path}, other parameters will have no effect')
            self.model = joblib.load(open(pretrained_path, 'rb'))
        else:
            self.model = KMeans(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                batch_size=batch_size,
                tol=tol,
                max_no_improvement=max_no_improvement,
                n_init=n_init,
                reassignment_ratio=reassignment_ratio,
                random_state=random_state,
                verbose=1,
                compute_labels=True,
                **kwargs
            )

    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.__repr__()

    def fit(self, data):
        '''
            data: [#B, #F] 
        '''
        print('Begin to fit')
        begin = stdtime()
        self.model.fit(data)
        end = stdtime()
        print(f'Fit ends, time: {(end-begin)/60:.2f} mins')

    def compute_inertia(self, data):
        '''
            data: [B, F]
        '''
        inertia = -self.model.score(data) / len(data)
        inertia = round(inertia, 2)
        return inertia
        
    def save(self, path):
        joblib.dump(self.model, open(path, 'wb'))


    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
        
