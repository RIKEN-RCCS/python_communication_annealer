# process.pngのみ(ビジュアライズできない)
import numpy as np
import networkx as nx
from common import main, Graphmodel
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import math

class Mesh6Dmodel(Graphmodel):
	def __init__(self,N,M,L,filename):
		self.N = N
		self.M = M
		self.L = L
		self.SIZE = 12 * N * M * L
		super().__init__(self.SIZE)
		self.visualizer = VisualizerVoid(self,filename)

	def generate_A(self):
		N=self.N
		M=self.M
		L=self.L
		A = np.array([[0 for _ in range(12 * N * M * L)] for _ in range(12 * N * M * L)])

		for i in range(L):
			for j in range(N*M):
				for m in range(12):
					k=12*(i*N*M+j)+m
					if (j+M)<N*M:
						A[k][k+12*M]=A[k+12*M][k]=1
					else:
						A[k][k-(N-1)*12*M]=A[k-(N-1)*12*M][k]=1
					if ((j%M)+1)<M:
						A[k][k+12]=A[k+12][k]=1
					else:
						A[k][k-12*M+12]=A[k-12*M+12][k]=1
					A[k][(k+12*N*M)%(12*N*M*L)]=A[(k+12*N*M)%(12*N*M*L)][k]=1

				for b in range(3):
					for n in range(2*2):
						c=12*(i*N*M+j)
						k=b*2*2+n
						if (n+2)<2*2:
							A[c+k][c+k+2]=A[c+k+2][c+k]=1
						else:
							A[c+k][c+k-(2-1)*2]=A[c+k-(2-1)*2][c+k]=1
						if ((n%2)+1)<2:
							A[c+k][c+k+1]=A[c+k+1][c+k]=1
						else:
							A[c+k][c+k-2+1]=A[c+k-2+1][c+k]=1
						A[c+k][c+(k+2*2)%(2*2*3)]=A[c+(k+2*2)%(2*2*3)][c+k]=1
		
		return A

	def generate_pos(self,G,perm):
		pass

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
				# f = 1 # 縦横を優先する変数 0:たて,1:よこ,2:高さ
				while t != s:
					next = (np.inf,-1)
					for j in range(SIZE):
						if (A[j][t] == 1) and (d[j]+1 == d[t]):
							next = min(next,j) # 仮
					if next[0] == np.inf:
						print("FAILED")
						sys.exit(1)
					prev = next[1]
					cost[t][prev] += C[s][i]
					cost[prev][t] += C[s][i]
					t = prev
					# f = (f + 1) % 3

			return cost

		dis = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])
		for i in range(SIZE):
			dis += cal_cost(i)

		return dis


class VisualizerVoid:
	def __init__(self,model,filename):
		self.model = model
		self.filename = filename
		self.red = -1
		self.green = -1
		self.sum = -1

	def generate_graph(self,filename,score,perm,A,C):
		pass


if __name__ == "__main__":
	N,M,L = 3,3,3
	SIZE = 12 * N * M * L
	steps = 10000
	filename = "tofu"

	rng = np.random.default_rng()
	Rand = [[rng.integers(0,100).item() for _ in range(SIZE)] for _ in range(SIZE)]
	C = (np.tril(Rand) + np.tril(Rand).T - 2*np.diag(np.diag(Rand))).astype(np.int64) # 対角成分が0であるトラフィック量を表す対称行列C

	main(lambda:Mesh6Dmodel(N,M,L,filename),SIZE,steps,filename,C)