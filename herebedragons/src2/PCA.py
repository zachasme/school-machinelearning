import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class PCA:
	"""Computes the principal components"""
	#rho
	#Z

	def __init__(self, data_set):
		"""Performs PCA analysis"""
		X     = data_set.X
		Y     = X - X.mean(0)
		U,S,V = np.linalg.svd(Y, full_matrices=False)
		print(S)
		# computes variance explained by principal components
		self.rho = pd.Series((S*S) / (S*S).sum())
		# projects the centered data onto principal component space, Z
		V = V.T
		self.Z = Y.dot(V)

	def plot_rho(self):
		plt.figure()		
		cumsum = self.rho.cumsum()
		cumsum.plot()
		plt.show()

	def show(self):
		i = 0
		j = 1
		plt.figure()

		x = self.Z.iloc[:,i]
		y = self.Z.iloc[:,j]

		plt.scatter(x, y, label="TODO:LABEL?")
		plt.show()


if __name__ == "__main__":
    import main