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
			df.fillna(df.mean(), inplace=True)
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



if __name__ == "__main__":
    import main