import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class DataSet:
	@property
	def X(self):
		"""The Data matrix (N x M numpy.matrix)
		The rows correspond to N data objects, each of which contains M attributes"""
		if self._class_column is None:
			dataframe = self.df
		else:
			dataframe = self.df.drop(self._class_column, axis=1)
		return dataframe.as_matrix()


	def __init__( self, datafile=None, na_values=[], dataframe=None, string_columns=[], class_column=None ):
		"""Creates the data set"""

		# if filepath is given, read as csv
		if datafile is not None:
			self.df = pd.read_csv(datafile, na_values=na_values)
		# else, if dataframe is given, use that
		elif dataframe is not None:
			self.df = dataframe

		# convert string columns to indices
		for c in string_columns:
			if not c in self.df.columns:
				warnings.warn("Column " + c + "given in string_columns, but does not exist in data")
			else:
				self.df[c] = self.df[c].apply( lambda x: np.nan if x is np.nan else self.df[c].tolist().index(x) )

		self._class_column = class_column

		# warnings
		for c in string_columns:
			if not c in self.df.columns:
				warnings.warn("Column " + c + "given in string_columns, but does not exist in data")

	def _copy(self, dataframe=None, class_column=None ):
		"""Creates a new dataset from dataframe but with same internal attributes"""
		if dataframe is None:
			dataframe = self.df
		if class_column is None:
			class_column = self._class_column

		return DataSet( dataframe=dataframe, class_column=class_column )




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
		return self.df[self._class_column].unique()

	@property
	def y(self):
		"""class index, a (Nx1) matrix.
		   for each data object, y contains a class index,
		   y in {0,1,...,C-1} where C is number of classes"""
		if self._class_column is None:
			raise Exception("DataSet: reading y property, but class-column not set")

		return self.df[self._class_column].apply(lambda x: np.nan if x is np.nan else self.classNames.tolist().index(x)).as_matrix()



	def classIn(self, class_column):
		return self._copy( class_column=class_column )



	def drop_columns(self, columns):
		dataframe = self.df.drop(columns, axis=1)

		return self._copy( dataframe=dataframe )

	def take_columns(self, columns):
		dataframe = self.df[columns]

		return self._copy( dataframe=dataframe )



	def normalize(self):
		"""Rescales (non-nominal) attributes to lie within interval [0,1]"""
		dataframe = ( self.df - self.df.min() ) / ( self.df.max() - self.df.min() )
		return self._copy( dataframe=dataframe )

	def standardize(self):
		"""Scales (non-nominal) data to zero mean (sigma=0) and unit variance (std=1)"""
		dataframe = (self.df - self.df.mean()) / self.df.std()

		return self._copy( dataframe=dataframe )


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



	def discretize(self, column, bins):
		bins = pd.cut(self.df[column], bins)
		self.df[column] = bins

		return self._copy( )

	def binarize(self, column, bins=1):
		# creates binary attributes
		cols = pd.crosstab(self.df.index, self.df[column])
		# prepends column name to interval labels
		cols = cols.rename(columns=lambda x: str(column) + str(x))

		# removes old column
		dataframe = self.drop_columns([column])

		# joins new columns to dataframe
		return dataframe._copy( dataframe=cols.join(dataframe.df) )



	def __repr__(self):
		return str(self.df)