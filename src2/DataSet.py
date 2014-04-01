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
		"""data matrix, rows correspond to N data objects, each of which contains M attributes"""
		return self.__df

	@property
	def N(self):
		"""Number of data objects"""
		(N, _) = self.__df.shape
		return N

	@property
	def M(self):
		"""Number of attributes"""
		(_, M) = self.__df.shape
		return M

	@property
	def C(self):
		"""Number of classes"""
		return len(self.classNames)

	@property
	def attributeNames(self):
		""""""
		return self.__df.columns.values

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



	def __init__(self, data, names, **options):
		self.options = {
			'drop_columns': options.get('drop_columns', []),
			'fix_missing':  options.get('fix_missing', False),
			'rescale':      options.get('rescale', False),

			'class_column': options.get('class_column', False),
		}

		"""Create the data set and preprocess"""
		self.__df = pd.DataFrame(data, columns=names)

		if self.options['class_column'] != False:
			label = pd.Categorical.from_array(self.__df[self.options['class_column']])
			self.__df['y'] = label.labels
			self.__classNames = label

		self.__preprocess();

		self.__y = self.__df['y'].values
		self.__df.drop(['y'], axis=1, inplace=True)






	def __preprocess(self):
		# drops unwanted columns
		self.__df.drop(self.options['drop_columns'], axis=1, inplace=True)

		# fixes missing values
		if self.options['fix_missing'] == FixMissing.FILLMEAN:
			self.__df.fillna(self.__df.mean(), inplace=True)
		if self.options['fix_missing'] == FixMissing.DROPOBJECTS:
			self.__df.dropna(axis=0, inplace=True);
		if self.options['fix_missing'] == FixMissing.DROPATTRIBUTES:
			self.__df.dropna(axis=1, inplace=True);

		y = self.__df['y']
		# rescales data by normalization or standardization
		if self.options['rescale'] == Rescale.NORMALIZE:
			"""Rescale attributes to lie within interval [0,1]"""
			self.__df = (self.__df - self.__df.min()) / (self.__df.max() - self.__df.min())
		if self.options['rescale'] == Rescale.STANDARDIZE:
			"""Scales data to zero mean (sigma=0) and unit variance (std=1)"""
			self.__df = (self.__df - self.__df.mean()) / self.__df.std();
		self.__df['y'] = y



if __name__ == "__main__":
    import main