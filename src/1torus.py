import numpy as np
import networkx as nx
from common import main, Graphmodel
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import math

class Torus1Dmodel(Graphmodel):
	def __init__(self,N,filename):
		self.N = N
		self.SIZE = N
		super().__init__(self.SIZE)
		self.visualizer = Visualizer1D(self,filename)

	def generate_A(self):
		SIZE = self.size
		A = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])

		for i in range(SIZE):
			A[i][(i+1)%(SIZE)]=A[(i+1)%(SIZE)][i]=1
		return A

	def generate_pos(self,G,perm):
		pos1 = nx.kamada_kawai_layout(G)
		return pos1

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
				while t != s:
					next = np.inf
					for j in range(SIZE):
						if (A[j][t] == 1) and (d[j]+1 == d[t]):
							next = min(next,perm_inv[j])
					if next == np.inf:
						print("FAILED")
						sys.exit(1)
					prev = perm[next]
					cost[t][prev] += C[s][i]
					cost[prev][t] += C[s][i]
					t = prev

			return cost

		dis = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])
		for i in range(SIZE):
			dis += cal_cost(i)

		return dis


class Visualizer1D:
	def __init__(self,model,filename):
		self.model = model
		self.filename = filename
		self.red = -1
		self.green = -1
		self.sum = -1

	def generate_graph(self,filename,score,perm,A,C):
		SIZE = self.model.SIZE

		plt.figure()
		plt.suptitle(str(score),fontsize=20)
		G = nx.Graph()
		dis = self.model.calculate_cost(A,C,perm)

		count = np.count_nonzero(dis)
		if self.red== -1:
			self.sum = np.sum(dis)
			self.red = sorted(dis.ravel())[-int(0.3*count)] # 要調整
			self.green = sorted(dis.ravel())[int(0.3*count+(SIZE*SIZE-count))] # 要調整
		
		edgecolor = []
		width = []
		all = 0
		nmax = 0 
		for i in range(SIZE):
			G.add_node(i)

		for i in range(SIZE):
			for j in range(i+1,SIZE):
				if A[i][j] == 1:
					G.add_edge(i,j)
					if self.red < dis[i][j]:
						edgecolor.append("Red")
					elif dis[i][j] < self.green:
						edgecolor.append("Green")
					else:
						edgecolor.append("Black")
					width.append(dis[i][j]/(self.sum/SIZE)*7)
					all += dis[i][j]
					nmax = max(nmax,dis[i][j])

		print("\nsum: "+str(all))
		print("max: "+str(nmax)+"\n")
		pos = self.model.generate_pos(G,perm)

		nx.draw(G,width=width,edge_color=edgecolor,pos=pos,with_labels=True)
		plt.savefig(f"{filename}.png")
		plt.close()
		# os.system(f"img2sixel {filename}.png")


if __name__ == "__main__":
	N = 16
	SIZE = N 
	steps = 100000
	filename = "1torus"

	rng = np.random.default_rng()
	Rand = [[rng.integers(0,100).item() for _ in range(SIZE)] for _ in range(SIZE)]
	C = (np.tril(Rand) + np.tril(Rand).T - 2*np.diag(np.diag(Rand))).astype(np.int64) # 対角成分が0であるトラフィック量を表す対称行列C

	main(lambda:Torus1Dmodel(N,filename),SIZE,steps,filename,C)