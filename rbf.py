import numpy as np
import math

class RBF():
	def __init__(self, numIn, numHidden, numOut):
		self.numIn = numIn 
		self.numHidden = numHidden
		self.numOut = numOut
		self.center = []
		self.beta = []
		self.on = []
		self.ofn = []
		self.wo = np.random.uniform(-0.5, 0.5, (self.numOut, self.numHidden + 1))

	def findCenter(self, t1, t2, t3):
		center1 = np.zeros(len(t1[0]))
		center2 = np.zeros(len(t2[0]))
		center3 = np.zeros(len(t3[0]))

		for i in range(len(t1)):
			for j in range(len(t1[0])):
				center1[j] += t1[i][j]

		center1 = center1/len(t1)

		for i in range(len(t2)):
			for j in range(len(t2[0])):
				center2[j] += t2[i][j]

		center2 = center2/len(t2)

		for i in range(len(t3)):
			for j in range(len(t3[0])):
				center3[j] += t3[i][j]

		center3 = center3/len(t3)

		self.center.append(center1.tolist())
		self.center.append(center2.tolist())
		self.center.append(center3.tolist())
		self.center = np.asarray(self.center)

		sigma = (1/len(t1))*np.sum(np.sqrt(np.sum((t1 - center1) **2, axis=1)))
		beta = 1/(2*(sigma**2))
		self.beta.append(beta)
		sigma = 0
		beta = 0

		sigma = (1/len(t2))*np.sum(np.sqrt(np.sum((t2 - center2) **2, axis=1)))
		beta = 1/(2*(sigma**2))
		self.beta.append(beta)
		sigma = 0
		beta = 0

		sigma = (1/len(t3))*np.sum(np.sqrt(np.sum((t3 - center3) **2, axis=1)))
		beta = 1/(2*(sigma**2))
		self.beta.append(beta)
		self.beta = np.asarray(self.beta)

	def activationFunciton(self, inputs, center, beta):
		return (np.exp(-beta*np.sum((inputs-center)**2, axis=1)))

	def feed(self, inputs):
		self.ho = self.activationFunciton(inputs, self.center, self.beta)
		hfn = self.ho.copy()
		hfn = np.append(hfn, 1)
		ow = np.multiply(self.wo, hfn)
		self.on = []
		self.ofn = []

		for i in range(ow.shape[0]):
			self.on.append(np.sum(ow[i]))
			self.ofn.append(self.on[i])

	def train(self, dataSet, out, eta=0.3, maxIterations=1000):
		t1, t2, t3 = normalize(dataSet) #Normaliza os dados
		x.findCenter(t1, t2, t3) #Encontra os centros

		tot = []
		tot.append(t1.tolist())
		tot.append(t2.tolist())
		tot.append(t3.tolist())

		squareError = 0
		it = 0
		while(it < maxIterations):
			squareError = 0
			k = 0

			for i in range(len(tot)):
				for j in range(0, 70):
					xi = tot[i][j]
					yi = out[k]
					k += 1

					self.feed(xi)

					error = np.array(yi) - np.array(self.ofn)
					squareError += np.sum(np.power(error, 2))
					aux = eta*np.dot(np.transpose(np.matrix(error)), np.matrix(np.append(self.ho, 1)))
					self.wo += aux

			squareError = squareError/len(dataSet)
			it += 1

	def test(self, dataSet):
		t1, t2, t3 = normalize(dataSet)
		print("Resultado: RBF")
		x = 0

		for i in range(0, len(t3)):
			self.feed(t3[i][2])
			x += self.ofn[2]

		print(x/len(t3)) 

def findMax(t1):
	maxValue = 0
	for i in range(len(t1)):
		if(max(t1[i]) > maxValue):
			maxValue = max(t1[i])

	t1 = np.array(t1[:])/maxValue

	return t1 

def normalize(data):
	print("normalizando")
	t1 = []
	t2 = []
	t3 = []

	for i in range(len(data)):
		if(data[i][1] == 1):
			t1.append(data[i][0])
		
		if(data[i][1] == 2):
			t2.append(data[i][0])

		if(data[i][1] == 3):
			t3.append(data[i][0])

	t1 = findMax(t1)
	t2 = findMax(t2)
	t3 = findMax(t3)

	return t1, t2, t3
if __name__ == '__main__':
	data = np.loadtxt("seeds.txt")
	dataSet = []
	for i in range(len(data)):
		y = np.array(data[:][i][:7])
		l = y.tolist()
		w = [l, data[:][i][7]]
		dataSet.append(w)
	
	v = []
	out = []
	for i in range(0, 210):
		if(i <= 69):
			v = [0.33 ,0,0]
		elif(i > 69 and i <= 139):
			v = [0, 0.66, 0]
		elif(i > 69 and i > 139 and i <= 209):
			v = [0, 0, 1]
		out.append(v)

	x = RBF(7, 3, 3)
	x.train(dataSet, out)
	x.test(dataSet)