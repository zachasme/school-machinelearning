import numpy as np
import pandas as pd

import matplotlib.pyplot as plt




#from enum import Enum
#FixMissing = Enum('FixMissing', 'FILLMEAN DROPOBJECTS DROPATTRIBUTES')
#Rescale    = Enum('Rescale',    'NORMALIZE STANDARDIZE')
class FixMissing:
	FILLMEAN = 0
	DROPOBJECTS = 1
	DROPATTRIBUTES = 2
class Rescale:
	NORMALIZE = 0
	STANDARDIZE = 1






class DataSet:

	@property
	def X(self):
		"""Data matrix

		type: numpy.matrix
		size: N x M

		The rows correspond to N data objects, each of which contains M attributes"""
		return self.__df

	@property
	def attributeNames(self):
		"""Attribute names

		type: list
		size: M x 1

		Name (string) for each of the M attributes"""
		return self.__df.columns.values

	@property
	def N(self):
		"""number of data objects
		an integer"""
		(N, _) = self.__df.shape
		return N

	@property
	def M(self):
		"""Number of attributes"""
		(_, M) = self.__df.shape
		return M

	# Regression/classification/cross-validation

	@property
	def C(self):
		"""Number of classes"""
		return len(self.classNames)



	@property
	def classNames(self):
		"""a Cx1 matrix of class names"""
		return self.__classNames

	@property
	def y(self):
		"""class index, a (Nx1) matrix.
		   for each data object, y contains a class index,
		   y in {0,1,...,C-1} where C is number of classes"""
		return self.__y



	def __init__(self, datafile, **options):
		self.options = {
			'fix_missing':  options.get('fix_missing', False),
			'rescale':      options.get('rescale', False),
		}

		"""Create the data set and preprocess"""
		self.__df = pd.read_csv(datafile, na_values=['?'])

		self.__preprocess();



	def __preprocess(self):
		# fixes missing values
		if self.options['fix_missing'] is FixMissing.FILLMEAN:
			self.__df.fillna(self.__df.mean(), inplace=True)
		if self.options['fix_missing'] is FixMissing.DROPOBJECTS:
			self.__df.dropna(axis=0, inplace=True);
		if self.options['fix_missing'] is FixMissing.DROPATTRIBUTES:
			self.__df.dropna(axis=1, inplace=True);

		# rescales data by normalization or standardization
		if self.options['rescale'] is Rescale.NORMALIZE:
			"""Rescale attributes to lie within interval [0,1]"""
			self.__df = (self.__df - self.__df.min()) / (self.__df.max() - self.__df.min())
		if self.options['rescale'] is Rescale.STANDARDIZE:
			"""Scales data to zero mean (sigma=0) and unit variance (std=1)"""
			self.__df = (self.__df - self.__df.mean()) / self.__df.std();



	def drop(self, columns):
		self.__df.drop(columns, axis=1, inplace=True)


if __name__ == "__main__":
    import main