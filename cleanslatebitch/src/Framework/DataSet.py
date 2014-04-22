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





class DataSet:
	def __init__( self, datafile=None, dataframe=None, nominals=[] ):
		"""Creates the data set"""
		self._nominals = nominals

		if datafile is not None:
			self.df = pd.read_csv(datafile, na_values=['?'])
		elif dataframe is not None:
			self.df = dataframe

		self._df_nominals     = self.df[self._nominals]
		self._df_non_nominals = self.df[self.df.columns - self._nominals]

	def _copy(self, dataframe=None, nominals=None):
		"""Creates a new dataset from dataframe but with same internal attributes"""
		if dataframe is None:
			dataframe = self.df
		if nominals is None:
			nominals = self._nominals

		return DataSet( dataframe=dataframe, nominals=nominals)


	@property
	def X(self):
		"""Data matrix

		type: numpy.matrix
		size: N x M

		The rows correspond to N data objects, each of which contains M attributes"""
		return self.df.values

	@property
	def attributeNames(self):
		"""Attribute names

		type: list
		size: M x 1

		Name (string) for each of the M attributes"""
		return self.df.columns.values

	@property
	def N(self):
		"""number of data objects (integer)"""
		(N, _) = self.df.shape
		return N

	@property
	def M(self):
		"""Number of attributes (integer)"""
		(_, M) = self.df.shape
		return M

	# Regression/classification/cross-validation

	@property
	def C(self):
		"""Number of classes (integer)"""
		return len(self.classNames)

	@property
	def classNames(self):
		"""a Cx1 matrix of class names

		type: list
		size: C x 1
		"""
		return self.__classNames

	@property
	def y(self):
		"""class index, a (Nx1) matrix.
		   for each data object, y contains a class index,
		   y in {0,1,...,C-1} where C is number of classes"""
		return self.__y



	def drop_columns(self, columns):
		dataframe = self.df.drop(columns, axis=1)
		nominals  = [column for column in self._nominals if column not in columns]

		return self._copy( dataframe=dataframe, nominals=nominals )

	def drop_nominals(self):
		return self.drop_columns(self._nominals)



	def normalize(self):
		"""Rescales (non-nominal) attributes to lie within interval [0,1]"""
		df_non_nominals = ( self._df_non_nominals - self._df_non_nominals.min() ) / ( self._df_non_nominals.max() - self._df_non_nominals.min() )
		df = pd.concat([self._df_nominals, df_non_nominals], axis=1)

		return self._copy( dataframe=df )

	def standardize(self):
		"""Scales (non-nominal) data to zero mean (sigma=0) and unit variance (std=1)"""
		df_non_nominals = self.center() / self._df_non_nominals.std()
		df = pd.concat([self._df_nominals, df_non_nominals], axis=1)

		return self._copy( dataframe=df )

	def center(self):
		"""Centers data to zero mean (sigma=0)"""
		df_non_nominals = self._df_non_nominals - self._df_non_nominals.mean()
		df = pd.concat([self._df_nominals, df_non_nominals], axis=1)

		return self._copy( dataframe=df )


	def fix_missing(self, drop_objects=False, drop_attributes=False, fill_mean=False):
		"""fixes missing values"""
		if fill_mean:
			df = self.df.fillna(self.df.mean())
		elif drop_attributes:
			df = self.df.dropna(axis=0)
		elif drop_objects:
			df = self.df.dropna(axis=1)
		else:
			raise Exception("fixmissing takes drop_objects, drop_attributes or fill_mean")

		return self._copy( dataframe=df )



	def binarize(self, column, bins):
		bins = pd.cut(self.df[column], bins)

		cols = pd.crosstab(self.df.index, bins)
		cols.index.name = "LOL"
		return (cols)



	def __repr__(self):
		return str(self.df)