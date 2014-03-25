from pylab import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Oli:
	# X: data matrix, rows correspond to N data objects, each of which contains M attributes
	# N: number of data objects
	# M: number of attributes
# OTHER VARIABLE NAMES:
# y: class index, a (Nx1) matrix.
#      for each data object, y contains a class index, y in {0,1,...,C-1}
#      where C is number of classes
# classNames:     a Cx1 matrix
# C:              number of classes

	def __init__(self, data):
		self.X = pd.DataFrame(data)
		self.normalize()
		self.standardize();

	def mean(self):
		"""Calculates means along TODO:WHAT?"""
		return X.mean(0)[np.newaxis,:]

	def N(self):
		(N,M) = X.shape
		return N

	def M(self):
		(N,M) = X.shape
		return M

	def normalize(self):
		"""Adjust all values to be between 0 and 1"""
		max_values = self.X.max(axis=1)
		self.X = self.X / max_values[:, np.newaxis]

	def standardize(self):
		"""Make sure variance is 1"""
		self.X = (self.X - self.X_mean) / std(X)

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