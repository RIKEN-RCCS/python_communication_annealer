import numpy as np
import networkx as nx
from common import main, Graphmodel
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import math

class Mesh3Dmodel(Graphmodel):
	def __init__(self,N,M,L,filename):
		self.N = N
		self.M = M
		self.L = L
		self.SIZE = N * M * L
		super().__init__(self.SIZE)
		self.visualizer = Visualizer3D(self,filename)

	def generate_A(self):
		N = self.N
		M = self.M
		L = self.L
		SIZE = self.SIZE
		
		A = np.array([[0 for _ in range(N * M * L)] for _ in range(N * M * L)])

		for i in range(L):
			for j in range(N*M):
				k=i*N*M+j
				if (j+M)<N*M:
					A[k][k+M]=A[k+M][k]=1
				else:
					A[k][k-(N-1)*M]=A[k-(N-1)*M][k]=1
				if ((j%M)+1)<M:
					A[k][k+1]=A[k+1][k]=1
				else:
					A[k][k-M+1]=A[k-M+1][k]=1
				A[k][(k+N*M)%(N*M*L)]=A[(k+N*M)%(N*M*L)][k]=1

		return A

	def generate_pos(self,G,perm):
		N = self.N
		M = self.M
		L = self.L
		# pos1=nx.spring_layout(G,dim=3,seed=42)
		pos2 = {perm[i]: (i % N, (i // N) % M, (i // (N * M)) % L) for i in range(SIZE)}
		return pos2

	def calculate_cost(self,A,C,perm):
		SIZE = self.SIZE

		perm_inv = np.array([0 for _ in range(SIZE)])
		for i in range(SIZE):
			perm_inv[perm[i]]=i

		def cal_cost(s):
			d = [np.inf for _ in range(SIZE)]
			used = [False for _ in range(SIZE)]
			d[s] = 0

			while True:
				v = -1
				for u in range(SIZE):
					if not used[u] and (v == -1 or d[u] < d[v]):
						v = u
				if v == -1:
					break
				used[v] = True
				for u in range(SIZE):
					if (A[v][u] == 1) and (d[u] > d[v]+1):
						d[u] = d[v]+1

			cost = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])
			for i in range(SIZE):
				t = i
				f = 1 # 縦横を優先 0:たて,1:よこ,2:高さ？
				# print(f"start: {t} , end: {s}")
				while t != s:
					next = (np.inf,-1)
					for j in range(SIZE):
						if (A[j][t] == 1) and (d[j]+1 == d[t]):
							if ((f == 0) and ((abs(perm_inv[j]-perm_inv[t]) == self.M*(SIZE/self.M-1)) or (abs(perm_inv[j]-perm_inv[t]) == self.M))) or ((f == 1) and ((abs(perm_inv[j]-perm_inv[t]) == 1) or (abs(perm_inv[j]-perm_inv[t]) == self.M-1))) or ((f == 2) and ((abs(perm_inv[j]-perm_inv[t]) == self.N*self.M) or (abs(perm_inv[j]-perm_inv[t]) == (self.N-1)*self.M*self.L))):
								next = min(next,(0,j))
							else:
								next = min(next,(1,j))
					if next[0] == np.inf:
						print("FAILED")
						sys.exit(1)
					prev = next[1]
					cost[t][prev] += C[s][i]
					cost[prev][t] += C[s][i]
					# print(t,f,next)
					t = prev
					f = (f + 1) % 3

			return cost

		dis = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])
		for i in range(SIZE):
			dis += cal_cost(i)

		return dis


class Visualizer3D:
	def __init__(self,model,filename):
		self.model = model
		self.filename = filename
		self.red = -1
		self.green = -1
		self.sum = -1

	def generate_graph(self, filename, score, perm, A, C):
		pass

if __name__ == "__main__":
	N,M,L = 4,4,4
	SIZE = N * M * L
	steps = 100000
	filename = "3torus"

	rng = np.random.default_rng()
	Rand = [[rng.integers(0,100).item() for _ in range(SIZE)] for _ in range(SIZE)]
	C = (np.tril(Rand) + np.tril(Rand).T - 2*np.diag(np.diag(Rand))).astype(np.int64) # 対角成分が0であるトラフィック量を表す対称行列C

	main(lambda:Mesh3Dmodel(N,M,L,filename),SIZE,steps,filename,C)
