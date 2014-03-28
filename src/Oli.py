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

		"""Create the data set"""
		self.X = pd.DataFrame(data, columns=names)

		"""Drop columns"""
		self.X = self.X.drop(self.options['drop_columns'], axis=1)
		"""Preprocessing"""
		# Fix missing values
		if self.options['fix_missing'] != False:
			self.__fixmissing()
		# Normalize and standardize
		if self.options['rescale'] != False:
			self.__rescale()



	def __fixmissing(self):
		"""Fixes missing values. How depends on option"""
		if not isinstance( self.options['fix_missing'], FixMissing ):
			raise Exception("Option 'fix_missing' was given value " + str(self.options['fix_missing']) + ", but needs to be one of " + str(list(FixMissing)))

		if self.options['fix_missing'] == FixMissing.FILLMEAN:
			self.X.fillna(self.X.mean())
		if self.options['fix_missing'] == FixMissing.DROPOBJECTS:
			self.X = self.X.dropna(axis=0);
		if self.options['fix_missing'] == FixMissing.DROPATTRIBUTES:
			self.X = self.X.dropna(axis=1);



	def __rescale(self):
		"""Rescales data during preprocessing"""
		if not isinstance( self.options['rescale'], Rescale ):
			raise Exception("Option 'rescale' was given value " + str(self.options['rescale']) + ", but needs to be one of " + str(list(Rescale)))

		if self.options['rescale'] == Rescale.NORMALIZE:
			"""Rescale attributes to lie within interval [0,1]"""
			self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

		if self.options['rescale'] == Rescale.STANDARDIZE:
			"""Scales data to zero mean (sigma=0) and unit variance (std=1)"""
			self.X = (self.X - self.X.mean()) / self.X.std();



	def mean(self):
		"""Calculates means along TODO:WHAT?"""
		return X.mean(0)[np.newaxis,:]

	def N(self):
		(N,M) = X.shape
		return N

	def M(self):
		(N,M) = X.shape
		return M


	def prep_elim_attr(self):
		"""removes data objects with missing values"""
		self.X = ma.compress_rows(X)
	
	def prep_elim_objs(self):
		"""removes attributes with missing values"""
		self.X = ma.compress_cols(X)

	def prep_fill_mean(self):
		"""takes mean where masked value otherwise"""
		self.X = np.where(X.mask, X_mean, X)

	def PCA(self):
		"""performs PCA analysis"""
		# first, remove mean, center at 0,0
		Y = X - X_mean
		U,S,V = linalg.svd(Y,full_matrices=False)

		# computes variance explained by principal components
		rho = (S*S) / (S*S).sum() 
		cumrho = cumsum(rho)

		# projects the centered data onto principal component space, Z
		V = mat(V).T
		Z = Y * V



	def __str__(self):
		return str(self.X)


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