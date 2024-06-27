import xarray as xr
import numpy as np
import dask.array as da
import uuid
import sys
import os

import dask.diagnostics as dd
#from ..flat_estimators.wrappers import nan_classifier, rf_classifier, nan_regression
#from ..flat_estimators.einstein_elm import extreme_learning_machine
#from ..flat_estimators.einstein_epoelm import epoelm
from sklearn.decomposition import PCA
from collections.abc import Iterable
#from .tuning import DFS, get_score


########### FROM THE UTILITIES PYTHON NOTEBOOK

def shape(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	return X.shape[list(X.dims).index(x_lat_dim)], X.shape[list(X.dims).index(x_lon_dim)], X.shape[list(X.dims).index(x_sample_dim)], X.shape[list(X.dims).index(x_feature_dim)]

def check_dimensions(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is 4D, with Dimension Names as specified by x_lat_dim, x_lon_dim, x_sample_dim, and x_feature_dim"""
	assert 4 <= len(X.dims) <= 5, 'XCast requires a dataset to be 4-Dimensional'
	assert x_lat_dim in X.dims, 'XCast requires a dataset_lat_dim to be a dimension on X'
	assert x_lon_dim in X.dims, 'XCast requires a dataset_lon_dim to be a dimension on X'
	assert x_sample_dim in X.dims, 'XCast requires a dataset_sample_dim to be a dimension on X'
	assert x_feature_dim in X.dims, 'XCast requires a dataset_feature_dim to be a dimension on X'
    
def check_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X has coordinates named as specified by x_lat_dim, x_lon_dim, x_sample_dim, and x_feature_dim"""
	assert x_lat_dim in X.coords.keys(), 'XCast requires a dataset_lat_dim to be a coordinate on X'
	assert x_lon_dim in X.coords.keys(), 'XCast requires a dataset_lon_dim to be a coordinate on X'
	assert x_sample_dim in X.coords.keys(), 'XCast requires a dataset_sample_dim to be a coordinate on X'
	assert x_feature_dim in X.coords.keys(), 'XCast requires a dataset_feature_dim to be a coordinate on X'

def check_consistent(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X's Coordinates are the same length as X's Dimensions"""
	assert X.shape[list(X.dims).index(x_lat_dim)] == len(X.coords[x_lat_dim].values), "XCast requires a dataset's x_lat_dim coordinate to be the same length as its x_lat_dim dimension"
	assert X.shape[list(X.dims).index(x_lon_dim)] == len(X.coords[x_lon_dim].values), "XCast requires a dataset's x_lon_dim coordinate to be the same length as its x_lon_dim dimension"
	assert X.shape[list(X.dims).index(x_sample_dim)] == len(X.coords[x_sample_dim].values), "XCast requires a dataset's x_sample_dim coordinate to be the same length as its x_sample_dim dimension"
	assert X.shape[list(X.dims).index(x_feature_dim)] == len(X.coords[x_feature_dim].values), "XCast requires a dataset's x_feature_dim coordinate to be the same length as its x_feature_dim dimension"

def check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is an Xarray.DataArray"""
	assert type(X) == xr.DataArray, 'XCast requires a dataset to be of type "Xarray.DataArray"'

def check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X satisfies all conditions for XCAST"""
	check_dimensions(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_consistent(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	#check_transposed(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

often_used = { 
	'longitude': ['LONGITUDE', 'LONG', 'X', 'LON'],
	'latitude': ['LATITUDE', 'LAT', 'LATI', 'Y'],
	'sample': ['T', 'S', 'TIME', 'SAMPLES', 'SAMPLE', 'INITIALIZATION', 'INIT','D', 'DATE', "TARGET", 'YEAR', 'I', 'N'],
	'feature': ['M', 'MODE', 'FEATURES', 'F', 'REALIZATION', 'MEMBER', 'Z', 'C', 'CAT', 'NUMBER', 'V', 'VARIABLE', 'VAR', 'P', 'LEVEL'],
}

def guess_coords(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	ret = {'latitude': x_lat_dim, 'longitude': x_lon_dim, 'sample': x_sample_dim, 'feature': x_feature_dim}
	user_provided_labels = [ ret[i] for i in ret.keys() if ret[i] is not None]
	labels_on_x = list(X.dims)
	for label in user_provided_labels: 
		assert label in labels_on_x, 'user-provided dimension ({}) not found on data-array: {}'.format(label, labels_on_x)
	labels_on_x_minus_user_provided = [ label for label in labels_on_x if label not in user_provided_labels]
	dims_left_to_find = [i for i in ret.keys() if ret[i] is None]
	for label in labels_on_x_minus_user_provided:
		for dim in dims_left_to_find: 
			for candidate in often_used[dim][::-1]:
				if label.upper() == candidate: 
					ret[dim] = label
					dims_left_to_find.pop(dims_left_to_find.index(dim))
	assigned_labels = [ ret[i] for i in ret.keys() if ret[i] is not None]
	unassigned_labels = [ label for label in labels_on_x if label not in assigned_labels]
	if len(unassigned_labels) == 1 and len(dims_left_to_find) == 1: 
		ret[dims_left_to_find[0]] = unassigned_labels[0]
		unassign_labels.pop(0)
		dims_left_to_find.pop(0)

	if len(unassigned_labels) > 0: 
		print('UNABLE TO ASSIGN FOLLOWING LABELS: {}'.format(unassigned_labels))
	if len(dims_left_to_find) > 0: 
		print('UNABLE TO FIND NAMES FOR FOLLOWING DIMS: {}'.format(dims_left_to_find))
	return ret['latitude'], ret['longitude'], ret['sample'], ret['feature']

def check_xyt_compatibility(X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None):
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	xlat, xlon, xsamp, xfeat = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
	ylat, ylon, ysamp, yfeat = shape(Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=y_feature_dim)

	assert xlat == ylat, "XCAST model training requires X and Y to have the same dimensions across XYT - latitude mismatch"
	assert xlon == ylon, "XCAST model training requires X and Y to have the same dimensions across XYT - longitude mismatch"
	assert xsamp == ysamp, "XCAST model training requires X and Y to have the same dimensions across XYT - sample mismatch"
    
    
########### FROM THE CHUNKING.PY SCRIPT

def align_chunks(X, Y, lat_chunks=5, lon_chunks=5, x_lat_dim=None, x_lon_dim=None, y_lat_dim=None, y_lon_dim=None, x_feature_dim=None, y_feature_dim=None, x_sample_dim=None, y_sample_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
        Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
    check_all(Y, y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim)

    x_lat_shape, x_lon_shape, x_samp_shape, x_feat_shape = shape(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_shape, y_lon_shape, y_samp_shape, y_feat_shape = shape(
        Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    X1 = X.chunk({x_lat_dim: max(x_lat_shape // lat_chunks, 1),
                 x_lon_dim: max(x_lon_shape // lon_chunks, 1)})
    Y1 = Y.chunk({y_lat_dim: max(y_lat_shape // lat_chunks, 1),
                 y_lon_dim: max(y_lon_shape // lon_chunks, 1)})

    X1 = X1.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    Y1 = Y1.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    return X1, Y1

################### FROM

class rf_classifier:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, x, y):
        self.model.fit(x, np.argmax(y, axis=1))
        self.classes = [i for i in range(y.shape[1])]

    def predict_proba(self, x):
        ret = self.model.predict_proba(x)
        for i in self.classes:
            if i not in self.model.classes_:
                ret = insert_zeros(ret, i)
        return ret

    def predict(self, x):
        ret = self.predict_proba(x)
        return np.argmax(ret, axis=1)


class nan_classifier:
    def __init__(self, **kwargs):
        self.model = None
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def fit(self, x, *args, **kwargs):
        self.x_features = x.shape[1]
        if len(args) > 0:
            y = args[0]
            self.y_features = y.shape[1]

    def transform(self, x, **kwargs):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], self.n_components))
        ret[:] = np.nan
        return ret

    def predict(self, x, n_out=1, **kwargs):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], n_out))
        ret[:] = np.nan
        return ret

    def predict_proba(self, x, n_out=3, **kwargs):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], n_out))
        ret[:] = np.nan
        return ret
    
class naive_bayes_classifier:
    def __init__(self, **kwargs):
        self.model = MultinomialNB(**kwargs)

    def fit(self, x, y):
        self.model.partial_fit(x, np.argmax(y, axis=-1),
                               classes=[i for i in range(y.shape[1])])

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        ret = self.model.predict_proba(x)
        return np.argmax(ret, axis=1)

class nan_regression:
    def __init__(self, **kwargs):
        self.model = None

    def fit(self, x, y=None):
        self.x_features = x.shape[1]

    def transform(self, x):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], 1))
        ret[:] = np.nan
        return ret

    def predict(self, x):
        assert self.x_features == x.shape[1]
        ret = np.empty((x.shape[0], 1))
        ret[:] = np.nan
        return ret
        
#####################
from scipy.special import expit, logit
import datetime as dt
import numpy as np
import scipy.linalg.lapack as la

class extreme_learning_machine:
    """Ensemble Extreme Learning Machine using Einstein Notation"""

    def __init__(self, activation='relu', hidden_layer_size=5, regularization=-10, preprocessing='minmax', n_estimators=30, eps=np.finfo('float').eps, activations=None):
        assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(hidden_layer_size)
        #assert type(c) is int, 'Invalid C {}'.format(c)
        assert type(preprocessing) is str and preprocessing in ['std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
        self.activation = activation
        self.regularization = 2**regularization if regularization is not None else regularization
        self.hidden_layer_size = hidden_layer_size
        self.preprocessing = preprocessing
        self.n_estimators=n_estimators
        self.eps=eps
        self.activations = {
            'sigm': expit,
            'tanh': np.tanh,
            'relu': lambda ret: np.maximum(0, ret),
            'lin': lambda ret: ret
        } if activations is None else activations
        assert activation in self.activations.keys(), 'invalid activation function {}'.format(activation)

    def set_params(self, **params):
        for key in params.keys():
            setattr(self, key, params[key])
        return self

    def get_params(self, deep=False):
        return vars(self)

    def fit(self, x, y):
        x, y = x.astype(np.float64), y.astype(np.float64)

        # first, take care of preprocessing
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        # after transformation, check feature dim
        x_features, y_features = x.shape[1], y.shape[1]

        # now, initialize weights & do all repeated stochastic initializations
        self.w = np.random.randn(self.n_estimators, x_features, self.hidden_layer_size)
        self.b = np.random.randn(self.n_estimators, 1, self.hidden_layer_size)

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        hth = np.stack([h[i, :, :].T.dot(h[i,:,:]) for i in range(self.n_estimators)], axis=0)#np.einsum('kni,kin->knn', np.transpose(h, [0, 2, 1]), h)
        eye = np.zeros(hth.shape)
        np.einsum('jii->ji', eye)[:] = 1.0
        hth_plus_ic = hth + ( ( eye / self.regularization)  if self.regularization is not None else  0 )
        ht = np.einsum('kni,i...->kn...', np.transpose(h, [0, 2, 1]), y)

        # one-hot encode Y
        bn = np.quantile(y.squeeze(), (1/3.0))
        an =  np.quantile(y.squeeze(), (2/3.0))
        y_terc  = np.ones((y.shape[0], 3))
        y_terc[(y.ravel() > bn) & (y.ravel() <=an), 0] = 0
        y_terc[(y.ravel() > bn) & (y.ravel() <=an), 2] = 0
        y_terc[y.ravel() < bn, 2] = 0
        y_terc[y.ravel() < bn, 1] = 0
        y_terc[y.ravel() >= an, 0] = 0
        y_terc[y.ravel() >= an, 1] = 0
        y_terc -= self.eps
        y_terc = np.abs(y_terc)


        logs = logit(y_terc)
        #logs = logs - 0.333 * (logs.max() - logs.min())
        ht_logs = np.einsum('kni,ij->knj', np.transpose(h, [0, 2, 1]), logs)

        self.betas = []
        self.gammas = []
        for i in range(self.n_estimators):
            if x.dtype == np.float64 and y.dtype == np.float64:
                _, B, info = la.dposv(hth_plus_ic[i,:,:], ht[i,:,:])
            elif x.dtype == np.float32 and y.dtype == np.float32:
                _, B, info = la.sposv(hth_plus_ic[i,:,:], ht[i,:,:])
            else:
                assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                    x.dtype, y.dtype)
            if info > 0:
                hth_plus_ic = hth_plus_ic + np.triu(hth_plus_ic, k=1)
                B = np.linalg.lstsq(hth_plus_ic[i,:,:], ht[i,:,:], rcond=None)[0]
            self.betas.append(B)

            if x.dtype == np.float64 and y.dtype == np.float64:
                _, B, info = la.dposv(hth_plus_ic[i,:,:], ht_logs[i,:,:])
            elif x.dtype == np.float32 and y.dtype == np.float32:
                _, B, info = la.sposv(hth_plus_ic[i,:,:], ht_logs[i,:,:])
            else:
                assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                    x.dtype, y.dtype)
            if info > 0:
                hth_plus_ic = hth_plus_ic + np.triu(hth_plus_ic, k=1)
                B = np.linalg.lstsq(hth_plus_ic[i,:,:], ht_logs[i,:,:], rcond=None)[0]
            self.gammas.append(B)

        self.beta = np.stack(self.betas, axis=0)
        self.gamma = np.stack(self.gammas, axis=0)


    def predict(self, x, preprocessing='asis'):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        return np.einsum('kin,kn...->ki...', h, self.beta).mean(axis=0)

    def predict_proba(self, x, preprocessing='asis'):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        h = self.activations[self.activation]( np.einsum('ij,kjn->kin', x, self.w) + self.b )
        act = np.einsum('kin,knj->ij', h, self.gamma)
        ret = expit( act / self.n_estimators)
        return ret / ret.sum(axis=-1).reshape(-1,1)

########## FROM EPOELM
from scipy.special import expit, logit
import datetime as dt
import numpy as np
import scipy.linalg.lapack as la
from collections.abc import Iterable
import copy
import matplotlib.pyplot as plt
import dask

def find_b(x, y):
    if x.dtype == np.float64 and y.dtype == np.float64:
        _, B, info = la.dposv(x, y)
    elif x.dtype == np.float32 and y.dtype == np.float32:
        _, B, info = la.sposv(x, y)
    else:
        assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
            x.dtype, y.dtype)
    if info > 0:
        print('info > 0!')
        x = x + np.triu(x, k=1)
        B = np.linalg.lstsq(x, y, rcond=None)[0]
    return B

    
class epoelm:
    """Ensemble Extreme Learning Machine using Einstein Notation"""

    def __init__(self, encoding='nonexceedance', initialization='normal', activation='relu', hidden_layer_size=5, regularization=0, regularize_lambda=True, preprocessing='minmax', n_estimators=25, eps=np.finfo('float').eps, activations=None, quantiles=[0.2, 0.4, 0.6, 0.8], standardize_y=False, save_y=True):
        assert isinstance(hidden_layer_size, int) and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(hidden_layer_size)
        #assert type(c) is int, 'Invalid C {}'.format(c)
        assert type(preprocessing) is str and preprocessing in ['std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
        self.activation = activation
        #self.c = c
        self.save_y = save_y
        self.quantiles=quantiles
        self.regularize_lambda = regularize_lambda
        self.encoding=encoding
        assert encoding.lower() in ['nonexceedance', 'binary'], 'invalid encoding for epoelm - must be "nonexceedance" or "binary"'
        self.hidden_layer_size = hidden_layer_size
        self.preprocessing = preprocessing
        self.initialization = initialization
        self.n_estimators=n_estimators
        self.eps2 = eps
        self.eps=0.01
        #regularizations = [0, 0.000976562, 0.0078125, 0.0625, 0.25, 0.5, 1, 4, 16, 256, 1028  ]
        #regularization = max( regularization, 5)
        self.regularization = 2**regularization if regularization is not None else None  #regularizations[regularization] if r
        self.activations = {
            'sigm': expit,
            'tanh': np.tanh,
            'relu': lambda ret: np.maximum(0, ret),
            'lin': lambda ret: ret,
            'softplus': lambda ret: np.logaddexp(ret, 0),
            'leaky': lambda ret: np.where(ret > 0, ret, 0.1*ret ),
            'elu': lambda ret: np.where(ret > 0, ret, 0.1* (np.exp(ret) - 1) ),
        } if activations is None else activations
        self.standardize_y = standardize_y
        assert activation in self.activations.keys(), 'invalid activation function {}'.format(activation)

    def set_params(self, **params):
        for key in params.keys():
            setattr(self, key, params[key])
        return self

    def get_params(self, deep=False):
        return vars(self)

    def fit(self, x, y):
        if self.n_estimators == 0:
            self.n_estimators = 30
        x, y = x.astype(np.float64), y.astype(np.float64)
        # first, take care of preprocessing
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.mean) / self.std  # scales to std normal dist

        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x = ((x - self.min) / (self.max - self.min)) * 2 - 1  # scales to [-1, 1]

        self.ymean = y.mean()
        self.ystd = y.std()
        self.quantile_of_mean = (y < self.ymean).mean()
        if self.ystd <= np.finfo(float).eps**19:
            print(y)
        assert self.ystd > np.finfo(float).eps**19, 'standard deviation of y is too close to zero! ... use a drymask? {}'.format(y)

        if self.standardize_y:
            y = ( y - self.ymean) / self.ystd

        x_features, y_features = x.shape[1], y.shape[1]
        if self.initialization == 'normal':
            self.w = np.random.randn(self.n_estimators, x_features, self.hidden_layer_size)
            self.b = np.random.randn(self.n_estimators, 1, self.hidden_layer_size)
        elif self.initialization == 'uniform':
            self.w = np.random.rand(self.n_estimators*x_features*self.hidden_layer_size).reshape(self.n_estimators, x_features, self.hidden_layer_size) * 2 - 1
            self.b = np.random.rand(self.n_estimators*self.hidden_layer_size).reshape(self.n_estimators, 1, self.hidden_layer_size) * 2 - 1
        else:
            self.w = np.random.rand(self.n_estimators*x_features*self.hidden_layer_size).reshape(self.n_estimators, x_features, self.hidden_layer_size) * (2 / np.sqrt(x_features)) - (1 / np.sqrt(x_features))
            self.b = np.random.rand(self.n_estimators*self.hidden_layer_size).reshape(self.n_estimators, 1, self.hidden_layer_size) * (2 / np.sqrt(x_features)) - (1 / np.sqrt(x_features))

        act = np.einsum('ij,kjn->kin', x, self.w) + self.b
        h = self.activations[self.activation]( act )
        qtemplate = np.ones_like(h[:,:, 0], dtype=float)

        if self.save_y:
            self.y = y.copy()

        quantiles = [ np.nanquantile(y, i) for i in self.quantiles ] #, method='midpoint') for i in self.quantiles ]
        self.iqr = quantiles[-1] - quantiles[0]
        self.q33, self.q67 = np.nanquantile(y, 1.0/3.0), np.nanquantile(y, 2.0/3.0)#, method='midpoint'), np.nanquantile(y, 2.0/3.0, method='midpoint')
        #thresholds = [ ( i - self.ymean ) / self.ystd for i in quantiles] # need to check that ystd > 0!!!!
        thresholds = quantiles

        hqs, ts = [], []
        for i, q in enumerate(thresholds):
            hq = np.concatenate( [h, np.expand_dims(qtemplate * q, axis=-1) ], axis=2)
            hqs.append(hq)
            if self.encoding.lower() == 'binary':
                t = np.zeros_like(y, dtype=float)
                t[y <= quantiles[i]] = 1
            else:
                t =  quantiles[i] - y #/ (y.max() - y.min()))
            ts.append(t)
        hqs = np.concatenate(hqs, axis=1)
        ts = np.vstack(ts)

        if self.encoding.lower() == 'binary':
            ts = np.abs(ts - self.eps2)
            logs = logit(ts)
        else:
            logs = ts

        qhth = np.stack([hqs[i, :, :].T.dot(hqs[i,:,:]) for i in range(self.n_estimators)], axis=0)
        qeye = np.zeros(qhth.shape)
        np.einsum('jii->ji', qeye)[:] = 1.0
        if not self.regularize_lambda:
            qeye[:, -1, -1] = 0.0
        qhth_plus_ic = qhth + ( ( qeye * self.regularization ) if self.regularization is not None else 0 )

        qht_logs = np.einsum('kni,ij->knj', np.transpose(hqs, [0, 2, 1]), logs)
        self.gammas = [ find_b(qhth_plus_ic[i,:,:], qht_logs[i,:,:]) for i in range(self.n_estimators) ]#[dask.delayed(find_b)(qhth_plus_ic[i,:,:], qht_logs[i,:,:]) for i in range(self.n_estimators)]
        #self.gammas = dask.compute(*self.gammas)
        self.gamma = np.stack(self.gammas, axis=0)
        # enforce gamma > 0

        self.lambdas = self.gamma[:, -1, :].mean(axis=-1)
        gamma = self.gamma[self.lambdas > self.eps, :, :]
       # self.w = self.w[self.lambdas > np.finfo(float).eps**19, :, :]
       # self.b = self.b[self.lambdas > np.finfo(float).eps**19, :, :]
        self.n_estimators = gamma.shape[0]
        
def get_score(x, y):
    return kling_gupta_efficiency(x, y)

def DFS(x,y, queue_len=5, gene_set=None, generation_size=5, n_mutations=2, lag=10, tol=0.001, estimator=epoelm, scorer=get_score):
    assert gene_set is not None, 'Must provide "Gene-Set" for hyperparameter tuning'
    #initialize queue
    best_params= get_random_selection(gene_set)
    best_score = scorer(best_params, x, y, estimator=estimator )
    queue = [ get_random_selection(gene_set) for j in range(queue_len-1) ]
    scores = [ scorer(i, x, y, estimator=estimator) for i in queue]
    queue, scores = sort_queue_by_score(queue, scores)
    history = [-999 + i for i in range(lag)]
    count = 3
    while len(queue) > 0 and  np.abs(best_score - history[-(lag-1)]) > tol:
        #print("    N Queued: {:>02.2f}, BestScore: {:1.7f}, WorstQueued: {:1.7}, TotalTested: {:>04}".format(len(queue), best_score, scores[0], count), end='\n')
        current = queue.pop(-1)
        current_score = scores.pop(-1)
        if current_score > best_score and ~np.isnan(current_score) and current_score < 1:
            best_score = current_score
            best_params = current
        for i in range(generation_size):
            params = current.copy()
            kys = [key for key in params.keys()]
            for j in range(n_mutations):
                ky = kys[np.random.randint(len(kys))]
                params[ky] = gene_set[ky][np.random.randint(len(gene_set[ky]))]
            score = scorer(params, x, y, estimator)
            count += 1
            makes_it = False
            for s in scores:
                if score > s:
                    makes_it = True
            if makes_it:
                scores.append(score)
                queue.append(params)
            queue, scores = sort_queue_by_score(queue, scores)
            history.append(scores[0] if len(scores) > 0 else -999)
            n = history.pop(0)

        # prune queue
        while len(queue) > queue_len:
            queue.pop(0)
            scores.pop(0)
    return  best_params, best_score, 0


############### FROM BASE ESTIMATORS

def apply_tune_to_block(x_data, y_data, mme=epoelm, scorer=get_score, ND=1, queue_len=5, generation_size=5, n_mutations=2, lag=10, tol=0.001, kwargs={}):
    x_data2 = x_data.reshape(x_data.shape[0]*x_data.shape[1]*x_data.shape[2], x_data.shape[3] )
    y_data2 = y_data.reshape(y_data.shape[0]*y_data.shape[1]*y_data.shape[2], y_data.shape[3] )
    x_data2 = x_data2[~np.isnan(np.sum(y_data2, axis=1)) ]
    y_data2 = y_data2[~np.isnan(np.sum(y_data2, axis=1)) ]
    if y_data2.shape[0] > 0:
        params, score, _ = DFS(x_data2, y_data2, queue_len=queue_len, generation_size=generation_size, n_mutations=n_mutations, lag=lag, tol=tol,  estimator=mme, scorer=scorer, gene_set=kwargs)
    else:
        params, score = {}, np.nan
    #print(params, score)
    models = np.empty(
        (x_data.shape[0], x_data.shape[1], ND), dtype=np.dtype('O'))
    scores = np.empty(
        (x_data.shape[0], x_data.shape[1], 1), dtype=np.dtype('O'))
    params2 = np.empty(
        (x_data.shape[0], x_data.shape[1], 1), dtype=np.dtype('O'))
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            y_train = y_data[i, j, :, :]
            if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
                temp_mme = nan_classifier
            else:
                temp_mme = mme
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            if len(y_train.shape) < 2:
                y_train = y_train.reshape(-1, 1)
            for k in range(ND):
                models[i][j][k] = temp_mme(**params)
                models[i][j][k].fit(x_train, y_train)
            scores[i][j][0] = score
            params2[i][j][0] = params
    return np.concatenate([scores, params2], axis=2)


def apply_fit_to_block(x_data, y_data, mme=epoelm, ND=1, kwargs={}):
    models = np.empty(
        (x_data.shape[0], x_data.shape[1], ND), dtype=np.dtype('O'))
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            y_train = y_data[i, j, :, :]
            if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
                temp_mme = nan_classifier
            else:
                temp_mme = mme
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            if len(y_train.shape) < 2:
                y_train = y_train.reshape(-1, 1)
            for k in range(ND):
                models[i][j][k] = temp_mme(**kwargs)
                models[i][j][k].fit(x_train, y_train)
    return models


def apply_hyperfit_to_block(x_data, y_data, models, mme=epoelm, ND=1, kwargs={}):
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            y_train = y_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            if len(y_train.shape) < 2:
                y_train = y_train.reshape(-1, 1)
            if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
                models[i][j] = nan_classifier()
            else:
                models[i][j] = mme(**models[i][j])
            models[i][j].fit(x_train, y_train)
    return models


def apply_fit_x_to_block(x_data, mme=PCA, ND=1, kwargs={}):
    models = np.empty(
        (x_data.shape[0], x_data.shape[1], ND), dtype=np.dtype('O'))
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            if np.isnan(np.min(x_train)):
                temp_mme = nan_classifier
            else:
                temp_mme = mme
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(ND):
                models[i][j][k] = temp_mme(**kwargs)
                models[i][j][k].fit(x_train)
    return models


def apply_predict_proba_to_block(x_data, models, kwargs={}):
    if 'n_out' in kwargs.keys():
        n_out = kwargs['n_out']
        kwargs = {k: v for k, v in kwargs.items() if k != 'n_out'}
    else:
        n_out = 3

    ret = np.empty((x_data.shape[0], x_data.shape[1],
                   models.shape[2], x_data.shape[2], n_out), dtype=float)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(models.shape[2]):
                if isinstance(models[i][j][k],  nan_classifier):
                    ret[i, j, k, :, :] = models[i][j][k].predict_proba(
                        x_train, n_out=n_out, **kwargs)
                else:
                    ret[i, j, k, :, :] = models[i][j][k].predict_proba(
                        x_train, **kwargs)
    return np.asarray(ret)


def apply_transform_to_block(x_data, models, kwargs={}):
    ret = []
    for i in range(x_data.shape[0]):
        ret.append([])
        for j in range(x_data.shape[1]):
            ret[i].append([])
            x_train = x_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(models.shape[2]):
                ret1 = models[i][j][k].transform(x_train, **kwargs)
                ret[i][j].append(ret1)
    return np.asarray(ret)


def apply_predict_to_block(x_data, models, kwargs={}):
    if 'n_out' in kwargs.keys():
        n_out = kwargs['n_out']
        kwargs = {k: v for k, v in kwargs.items() if k != 'n_out'}
    else:
        n_out = 1
    ret = np.empty((x_data.shape[0], x_data.shape[1],
                   models.shape[2], x_data.shape[2], n_out), dtype=float)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            x_train = x_data[i, j, :, :]
            if len(x_train.shape) < 2:
                x_train = x_train.reshape(-1, 1)
            for k in range(models.shape[2]):
                if isinstance(models[i][j][k],  nan_classifier):
                    ret1 = models[i][j][k].predict(
                        x_train, n_out=n_out, **kwargs)
                else:
                    ret1 = models[i][j][k].predict(x_train, **kwargs)
                if len(ret1.shape) < 2:
                    ret1 = np.expand_dims(ret1, axis=1)
                ret[i, j, k, :, :] = ret1
    return np.asarray(ret)


class BaseEstimator:
    """ BaseEstimator class
    implements .fit(X, Y) and, .predict_proba(X), .predict(X)
    can be sub-classed to extend to new statistical methods
    new methods must implement .fit(x, y) and .predict(x)
    and then sub-class's .model_type must be set to the constructor of the new method """

    def __init__(self, client=None, lat_chunks=1, lon_chunks=1, verbose=False, params=None, **kwargs):
        self.model_type = epoelm
        self.models_, self.ND = None, 1
        self.client, self.kwargs = client, kwargs
        self.verbose = verbose
        self.lat_chunks, self.lon_chunks = lat_chunks, lon_chunks
        self.latitude, self.longitude, self.features = None, None, None
        self.params = params.copy() if params is not None else None

    def fit(self, X, *args,  x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, rechunk=True):
        if len(args) > 0:
            assert len(args) < 2, 'too many args'
            Y = args[0]

            x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
                X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
                Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
            check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
            check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim,
                                    x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
            self.latitude, self.longitude, _, self.features = shape(
                X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

            if Y.dims[0] != y_lat_dim or Y.dims[1] != y_lon_dim or Y.dims[2] != y_sample_dim or Y.dims[3] != y_feature_dim:
                Y1 = Y.transpose(y_lat_dim, y_lon_dim,
                                 y_sample_dim, y_feature_dim)
            else:
                Y1 = Y

            if rechunk:
                X1, Y1 = align_chunks(X1, Y1,  self.lat_chunks, self.lon_chunks, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                                      x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)
            if self.params is not None:
                if self.params.dims[0] != x_lat_dim or self.params.dims[1] != x_lon_dim:
                    self.params= self.params.transpose(x_lat_dim, x_lon_dim)
                else:
                    self.params = self.params
                models_data = self.params.data

                x_data = X1.data
                y_data = Y1.data
                if not isinstance(x_data, da.core.Array):
                    x_data = da.from_array(x_data)
                if not isinstance(y_data, da.core.Array):
                    y_data = da.from_array(y_data, chunks=x_data.chunksize)
                if not isinstance(models_data, da.core.Array):
                    models_data = da.from_array(models_data, chunks=y_data.chunksize[:2])

                self.models_ = da.blockwise(apply_hyperfit_to_block, 'ij', x_data, 'ijkl', y_data, 'ijkm', models_data, 'ij', concatenate=True, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
                if type(self.models_) == np.ndarray:
                    self.models_ = da.from_array(self.models_, chunks=(max(
                        self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
                self.models_ = da.stack( [self.models_ for ijk in range(self.ND)], axis=-1)
                #self.models = xr.DataArray(name='models', data=self.models_, dims=[x_lat_dim, x_lon_dim], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values })

            else: 
                x_data = X1.data
                y_data = Y1.data
                if not isinstance(x_data, da.core.Array):
                    x_data = da.from_array(x_data)
                if not isinstance(y_data, da.core.Array):
                    y_data = da.from_array(y_data)

                if self.verbose:
                    with dd.ProgressBar():
                        self.models_ = da.blockwise(apply_fit_to_block, 'ijn', x_data, 'ijkl', y_data, 'ijkm', new_axes={
                            'n': self.ND}, mme=self.model_type, ND=self.ND, concatenate=True, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
                else:
                    self.models_ = da.blockwise(apply_fit_to_block, 'ijn', x_data, 'ijkl', y_data, 'ijkm', new_axes={
                        'n': self.ND}, mme=self.model_type, ND=self.ND, concatenate=True, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
                if type(self.models_) == np.ndarray:
                    self.models_ = da.from_array(self.models_, chunks=(max(
                        self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
                #self.models = xr.DataArray(name='models', data=self.models_, dims=[x_lat_dim, x_lon_dim, 'ND'], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values, 'ND': [iii+1 for iii in range(self.ND)]})
        else:
            x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
                X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            self.latitude, self.longitude, _, self.features = shape(
                X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

            X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

            x_data = X1.data
            if not isinstance(x_data, da.core.Array):
                x_data = da.from_array(x_data)

            if self.verbose:
                with dd.ProgressBar():
                    self.models_ = da.blockwise(apply_fit_x_to_block, 'ijn', x_data, 'ijkl', new_axes={
                        'n': self.ND}, mme=self.model_type, concatenate=True, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
            else:
                self.models_ = da.blockwise(apply_fit_x_to_block, 'ijn', x_data, 'ijkl',  new_axes={
                    'n': self.ND}, mme=self.model_type, concatenate=True, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()

            if type(self.models_) == np.ndarray:
                self.models_ = da.from_array(self.models_, chunks=(max(
                    self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
        self.models = xr.DataArray(name='models', data=self.models_, dims=[x_lat_dim, x_lon_dim, 'ND'], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values, 'ND': [iii+1 for iii in range(self.ND)]})


    def tune(self, X, Y,  x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, rechunk="1X1", queue_len=5, gene_set=None, generation_size=5, n_mutations=2, lag=10, tol=0.001, scorer=get_score, **kwargs):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim,
                                x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        self.latitude, self.longitude, _, self.features = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
            X1 = X.transpose(x_lat_dim, x_lon_dim,
                                x_sample_dim, x_feature_dim)
        else:
            X1 = X

        if Y.dims[0] != y_lat_dim or Y.dims[1] != y_lon_dim or Y.dims[2] != y_sample_dim or Y.dims[3] != y_feature_dim:
            Y1 = Y.transpose(y_lat_dim, y_lon_dim,
                                y_sample_dim, y_feature_dim)
        else:
            Y1 = Y

        if rechunk.upper() == "1X1":
            X1, Y1 = align_chunks(X1, Y1,  self.latitude, self.longitude, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                                    x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)
        elif rechunk:
            X1, Y1 = align_chunks(X1, Y1,  self.lat_chunks, self.lon_chunks, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                                    x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)
        else:
            pass

        x_data = X1.data
        y_data = Y1.data
        if not isinstance(x_data, da.core.Array):
            x_data = da.from_array(x_data)
        if not isinstance(y_data, da.core.Array):
            y_data = da.from_array(y_data)

        ret_= da.blockwise(apply_tune_to_block, 'ijn', x_data, 'ijkl', y_data, 'ijkm', new_axes={'n': 2}, mme=self.model_type, scorer=scorer, ND=self.ND, queue_len=queue_len, generation_size=generation_size, n_mutations=n_mutations, lag=lag, tol=tol, concatenate=True, kwargs=kwargs, meta=np.array((), dtype=np.dtype('O'))).persist()
        
        if type(self.models_) == np.ndarray:
            ret_ = da.from_array(ret_, chunks=(max(self.latitude // self.lat_chunks, 1), max(self.longitude // self.lon_chunks, 1), self.ND))
        scores = ret_[:,:,0].astype(float)
        params = ret_[:,:,1]
        scores = xr.DataArray(name='goodness', data=scores, dims=['latitude', 'longitude'], coords={'latitude': getattr(X1, x_lat_dim), 'longitude': getattr(X1, x_lon_dim)})
        params = xr.DataArray(name='params', data=params, dims=['latitude', 'longitude'], coords={'latitude': getattr(X1, x_lat_dim), 'longitude': getattr(X1, x_lon_dim)})
        return xr.merge([scores, params])


    def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True, **kwargs):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        xlat, xlon, xsamp, xfeat = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
        assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
        assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

        if 'n_out' not in kwargs.keys():
            if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
                if not isinstance(kwargs['quantile'], Iterable):
                    kwargs['quantile'] = [kwargs['quantile']]
                kwargs['n_out'] = len(kwargs['quantile'])
            elif 'threshold' in kwargs.keys() and kwargs['threshold'] is not None:
                if not isinstance(kwargs['threshold'], Iterable):
                    kwargs['threshold'] = [kwargs['threshold']]
                kwargs['n_out'] = len(kwargs['threshold'])
            else:
                kwargs['n_out'] = 3

        if rechunk:
            X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks, 1), x_lon_dim: max(xlon //
                         self.lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        else:
            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

        x_data = X1.data
        if self.verbose:
            with dd.ProgressBar():
                results = da.blockwise(apply_predict_proba_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                    'm': kwargs['n_out']},  dtype=float, concatenate=True, kwargs=kwargs).persist()
        else:
            results = da.blockwise(apply_predict_proba_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                    'm': kwargs['n_out']},  dtype=float, concatenate=True, kwargs=kwargs).persist()


        feature_coords = [i for i in range(kwargs['n_out'])]
        if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['quantile']
        if 'threshold' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['threshold']
        coords = {
            x_lat_dim: X1.coords[x_lat_dim].values,
            x_lon_dim: X1.coords[x_lon_dim].values,
            x_sample_dim: X1.coords[x_sample_dim].values,
            x_feature_dim:  feature_coords,
            'ND': [i for i in range(self.ND)]
        }

        dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
        attrs = X1.attrs
        attrs.update(
            {'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
        return xr.DataArray(name='predicted_probability', data=results, coords=coords, dims=dims, attrs=attrs).mean('ND')

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True, **kwargs):
        if 'n_out' not in kwargs.keys():
            assert 'n_components' in kwargs.keys(), 'if you dont pass n_components, you must pass n_out'
            kwargs['n_out'] = kwargs['n_components']
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        xlat, xlon, xsamp, xfeat = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
        assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
        assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

        if rechunk:
            X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks, 1), x_lon_dim: max(xlon //
                         self.lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        else:
            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

        x_data = X1.data
        if self.verbose:
            with dd.ProgressBar():
                results = da.blockwise(apply_transform_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                       'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()
        else:
            results = da.blockwise(apply_transform_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                   'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()

        coords = {
            x_lat_dim: X1.coords[x_lat_dim].values,
            x_lon_dim: X1.coords[x_lon_dim].values,
            x_sample_dim: X1.coords[x_sample_dim].values,
            x_feature_dim: [i for i in range(kwargs['n_out'])],
            'ND': [i for i in range(self.ND)]
        }

        dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
        attrs = X1.attrs
        attrs.update(
            {'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
        return xr.DataArray(name='transformed', data=results, coords=coords, dims=dims, attrs=attrs).mean('ND')

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True, **kwargs):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        xlat, xlon, xsamp, xfeat = shape(
            X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
        assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
        assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
        assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'
        if 'n_out' not in kwargs.keys():
            if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
                if not isinstance(kwargs['quantile'], Iterable):
                    kwargs['quantile'] = [kwargs['quantile']]
                kwargs['n_out'] = len(kwargs['quantile'])
            elif 'threshold' in kwargs.keys() and kwargs['threshold'] is not None:
                if not isinstance(kwargs['threshold'], Iterable):
                    kwargs['threshold'] = [kwargs['threshold']]
                kwargs['n_out'] = len(kwargs['threshold'])
            else:
                kwargs['n_out'] = 1
        if rechunk:
            X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks, 1), x_lon_dim: max(xlon //
                         self.lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        else:
            if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
                X1 = X.transpose(x_lat_dim, x_lon_dim,
                                 x_sample_dim, x_feature_dim)
            else:
                X1 = X

        x_data = X1.data
        if self.verbose:
            with dd.ProgressBar():
                results = da.blockwise(apply_predict_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                       'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()
        else:
            results = da.blockwise(apply_predict_to_block, 'ijnkm', x_data, 'ijkl', self.models_, 'ijn', new_axes={
                                   'm': kwargs['n_out']}, dtype=float, concatenate=True, kwargs=kwargs).persist()
        feature_coords = [i for i in range(kwargs['n_out'])]
        if 'quantile' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['quantile']
        if 'threshold' in kwargs.keys() and kwargs['quantile'] is not None:
            feature_coords = kwargs['threshold']
        coords = {
            x_lat_dim: X1.coords[x_lat_dim].values,
            x_lon_dim: X1.coords[x_lon_dim].values,
            x_sample_dim: X1.coords[x_sample_dim].values,
            x_feature_dim: feature_coords,
            'ND': [i for i in range(self.ND)]
        }

        dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
        attrs = X1.attrs
        attrs.update(
            {'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
        return xr.DataArray(name='predicted', data=results, coords=coords, dims=dims, attrs=attrs).mean('ND')

import copy
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
from collections.abc import Iterable

class extended_logistic_regression:
    def __init__(self, pca=-999, preprocessing='none', verbose=False, **kwargs):
        self.kwargs = kwargs
        self.an_thresh = 0.67
        self.bn_thresh = 0.33
        self.pca = pca
        self.preprocessing = preprocessing
        self.verbose = verbose

    def fit(self, x, y):
        self.models = [multivariate_extended_logistic_regression(
            pca=self.pca, preprocessing=self.preprocessing, verbose=self.verbose, **self.kwargs) for i in range(x.shape[1])]
        for i in range(x.shape[1]):
            self.models[i].bn_thresh = self.bn_thresh
            self.models[i].an_thresh = self.an_thresh
            self.models[i].fit(x[:, i].reshape(-1, 1), y)

    def predict_proba(self, x, quantile=None):
        res = []
        for i in range(x.shape[1]):
            res.append(self.models[i].predict_proba(
                x[:, i].reshape(-1, 1), quantile=quantile))
        res = np.stack(res, axis=0)
        return np.nanmean(res, axis=0)
    
    def predict(self, x, quantile=None):
        raise NotImplementedError


class multivariate_extended_logistic_regression:
    def __init__(self, pca=-999, preprocessing='none', verbose=False, thresholds=[0.33, 0.67], **kwargs):
        self.kwargs = kwargs
        self.thresholds = thresholds
        self.preprocessing = preprocessing
        self.verbose = verbose

    def fit(self, x, y):
        y = y[:, 0].reshape(-1, 1)
        # first, take care of preprocessing
        x2 = copy.deepcopy(x)
        y2 = copy.deepcopy(y)
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x2 = (copy.deepcopy(x) - self.mean) / \
                self.std  # scales to std normal dist
        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x2 = ((copy.deepcopy(x) - self.min) /
                  (self.max - self.min)) * 2 - 1  # scales to [-1, 1]
        self.y = y2
        bs = [np.quantile(y, thresh)#, method='midpoint')
                for thresh in self.thresholds]
        y = np.vstack([np.where(y < b, np.ones((y.shape[0], 1)).astype(
            np.float64)*0.999, np.ones((y.shape[0], 1)).astype(np.float64)*0.001) for b in bs])
        v = []
        for b in bs:
            x_bn = np.hstack(
                [x2, np.ones((x2.shape[0], 1), dtype=np.float64)*b])
            v.append(x_bn)
        x3 = np.vstack(v)
        model = sm.GLM(y, sm.add_constant(
            x3, has_constant='add'), family=sm.families.Binomial())
        self.model = model.fit()


    def nonexceedance(self, x, quantile=0.5):
        if not isinstance(quantile, Iterable):
            quantile = np.asarray([quantile])
        x2 = copy.deepcopy(x)
        if self.preprocessing == 'std':
            x2 = (copy.deepcopy(x) - self.mean) / \
                self.std  # scales to std normal dist
        if self.preprocessing == 'minmax':
            x2 = ((copy.deepcopy(x) - self.min) /
                  (self.max - self.min)) * 2 - 1  # scales to [-1, 1]
        thresh = np.quantile(self.y, quantile)#, method='midpoint')
        x_an = np.hstack([x2, np.ones((x.shape[0], 1)) * thresh])
        # self.x_an.append(x_an)
        x_an = sm.add_constant(x_an, has_constant='add')
        return self.model.predict(x_an).reshape(-1, 1)


    def exceedance(self, x, quantile=0.5):
        return 1 - self.nonexceedance(x, quantile=quantile)

    def predict_proba(self, x, quantile=None):
        if quantile is None:
            bn = self.nonexceedance(x, quantile=(1/3))
            an = self.exceedance(x, quantile=(2/3))
            nn = 1 - (bn + an)
            return np.hstack([bn, nn, an])
        else:
            if not isinstance(quantile, Iterable):
                quantile = np.asarray([quantile])
            return np.hstack([self.nonexceedance(x, quantile=q) for q in quantile])


    def predict(self, x, quantile=None):
        raise NotImplementedError

# from ..flat_estimators.elr import multivariate_extended_logistic_regression, extended_logistic_regression 
#from ..flat_estimators.wrappers import rf_classifier, naive_bayes_classifier
from sklearn.neural_network import MLPClassifier


class MELR(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = multivariate_extended_logistic_regression


class ELR(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = extended_logistic_regression
        
class EPOELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = epoelm