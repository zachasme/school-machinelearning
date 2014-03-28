from enum import Enum

import pandas as pd
import numpy as np

#from pylab import *
#import matplotlib.pyplot as plt


FixMissing = Enum('FixMissing', 'FILLMEAN DROPOBJECTS DROPATTRIBUTES')
Rescale    = Enum('Rescale',    'NORMALIZE STANDARDIZE')


class DataSet:
	# X: data matrix, rows correspond to N data objects, each of which contains M attributes
	# N: number of data objects
	# M: number of attributes
# OTHER VARIABLE NAMES:
# y: class index, a (Nx1) matrix.
#      for each data object, y contains a class index, y in {0,1,...,C-1}
#      where C is number of classes
# classNames:     a Cx1 matrix
# C:              number of classes

	def __init__(self, data, names, **options):
		self.options = {
		  'drop_columns': options.get('drop_columns', []),
			'fix_missing':  options.get('fix_missing', False),
			'rescale':      options.get('rescale', False),
		}

		"""Create the data set and preprocess"""
		df     = pd.DataFrame(data, columns=names)
		self.X = self.__preprocess(df);



	def __preprocess(self, df):
		# drops unwanted columns
		df = df.drop(self.options['drop_columns'], axis=1)

		# fixes missing values
		if self.options['fix_missing'] == FixMissing.FILLMEAN:
			df.fillna(df.mean())
		if self.options['fix_missing'] == FixMissing.DROPOBJECTS:
			df = df.dropna(axis=0);
		if self.options['fix_missing'] == FixMissing.DROPATTRIBUTES:
			df = df.dropna(axis=1);

		# rescales data by normalization or standardization
		if self.options['rescale'] == Rescale.NORMALIZE:
			"""Rescale attributes to lie within interval [0,1]"""
			df = (df - df.min()) / (df.max() - df.min())
		if self.options['rescale'] == Rescale.STANDARDIZE:
			"""Scales data to zero mean (sigma=0) and unit variance (std=1)"""
			df = (df - df.mean()) / df.std();

		return df



	def N(self):
		(N, _) = self.X.shape
		return N

	def M(self):
		(_, M) = self.X.shape
		return M



class PCA:
	#rho
	#Z

	def __init__(self, X):
		"""Performs PCA analysis"""
		Y     = X - X.mean
		U,S,V = linalg.svd(Y, full_matrices=False)

		# computes variance explained by principal components
		self.rho = (S*S) / (S*S).sum() 

		# projects the centered data onto principal component space, Z
		V = mat(V).T
		self.Z = Y * V

	def plot_rho(self):
		fig = figure()
		set_title("Variance explained as a function of number of PCs")
		set_xlabel("Number of principal components included")
		set_ylabel("Amount of variance explained")

		plot(cumsum(self.rho))

	def plot_components(self, i, j):
		fig = plt.figure()
		set_title("Plot of principal components "+i+" and "+j+".")
		set_xlabel("PCA #"+str(i+1))
		set_ylabel("PCA #"+str(j+1))

		x = self.Z[:,i].flat
		y = self.Z[:,j].flat

		scatter(x, y, label="TODO:LABEL?")


if __name__ == "__main__":
    import main