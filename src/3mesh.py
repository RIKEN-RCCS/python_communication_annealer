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

		A = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])

		for i in range(L):
			for j in range(N*M):
				k=i*N*M+j
				if (j+M)<N*M:
					A[k][k+M]=A[k+M][k]=1
				if ((j%M)+1)<M:
					A[k][k+1]=A[k+1][k]=1
				if (k+N*M)<(N*M*L):
					A[k][k+N*M]=A[k+N*M][k]=1

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
							if ((f == 0) and (abs(perm_inv[j]-perm_inv[t]) == self.M)) or ((f == 1) and (abs(perm_inv[j]-perm_inv[t]) == 1)) or ((f == 2) and (abs(perm_inv[j]-perm_inv[t]) == self.N*self.M)):
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

	def generate_graph(self,filename,score,perm,A,C):
		SIZE = self.model.SIZE

		plt.figure()
		plt.suptitle(score)
		G = nx.Graph()
		dis = self.model.calculate_cost(A,C,perm)
		count = np.count_nonzero(dis)
		if self.red== -1:
			self.sum = np.sum(dis)
			self.red = sorted(dis.ravel())[-int(0.3*count)] # 要調整
			self.green = sorted(dis.ravel())[int(0.3*count+(SIZE*SIZE-count))] # 要調整
		
		edgecolor = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
		width = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
		all = 0
		nmax = 0 
		for i in range(SIZE):
			G.add_node(i)

		for i in range(SIZE):
				for j in range(i+1,SIZE):
						if A[i][j]==1:
								G.add_edge(i,j)
								if self.red<dis[i][j]:
										edgecolor[i][j]=edgecolor[j][i]="Red"
								elif dis[i][j]<self.green:
										edgecolor[i][j]=edgecolor[j][i]="Green"
								else:
										edgecolor[i][j]=edgecolor[j][i]="Black"
								width[i][j]=width[j][i]=10*(dis[i][j]/(self.sum/SIZE)*7)
								all += dis[i][j]
								nmax = max(nmax,dis[i][j])

		print("\nsum: "+str(all))
		print("max: "+str(nmax)+"\n")
		pos = self.model.generate_pos(G,perm)

		xn=[pos[n][0] for n in G.nodes()]
		yn=[pos[n][1] for n in G.nodes()]
		zn=[pos[n][2] for n in G.nodes()]

		node_trace = go.Scatter3d(
			x=xn,y=yn,z=zn,
			mode='markers+text',
			text=[str(n) for n in G.nodes()],
			marker=dict(size=6,color='blue'),
			textposition='top center'
		)

		edge_traces=[]
		for u,v in G.edges():
			x=[pos[u][0],pos[v][0],None]
			y=[pos[u][1],pos[v][1],None]
			z=[pos[u][2],pos[v][2],None]

			edge_traces.append(go.Scatter3d(
				x=x,y=y,z=z,
				mode='lines',
				line=dict(color=edgecolor[u][v],width=width[u][v])
			))
		fig = go.Figure(data=[node_trace]+edge_traces)
		fig.update_layout(scene=dict(xaxis=dict(visible=False),
							yaxis=dict(visible=False),
							zaxis=dict(visible=False)))
		
		fig.write_html(f"{filename}.html")


if __name__ == "__main__":
	N,M,L = 4,4,4
	SIZE = N * M * L
	steps = 100000
	filename = "3mesh"

	rng = np.random.default_rng()
	Rand = [[rng.integers(0,100).item() for _ in range(SIZE)] for _ in range(SIZE)]
	C = (np.tril(Rand) + np.tril(Rand).T - 2*np.diag(np.diag(Rand))).astype(np.int64) # 対角成分が0であるトラフィック量を表す対称行列C
	
	main(lambda:Mesh3Dmodel(N,M,L,filename),SIZE,steps,filename,C)