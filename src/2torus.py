import numpy as np
import networkx as nx
from common import main, Graphmodel
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import math

class Torus2Dmodel(Graphmodel):
	def __init__(self,N,M,filename):
		self.N = N
		self.M = M
		self.SIZE = N * M
		super().__init__(self.SIZE)
		self.visualizer = Visualizer2D(self,filename)

	def generate_A(self):
		N,M = self.N,self.M
		SIZE = N * M
		A = np.array([[0 for _ in range(N * M)] for _ in range(N * M)])

		for i in range(N*M):
			A[i][(i+M)%(N*M)] = A[(i+M)%(N*M)][i] = 1
			if (i+1)%M == 0:
				A[i][i-M+1] = A[i-M+1][i] = 1
			else:
				A[i][i+1] = A[i+1][i] = 1

		return A

	def generate_pos(self,G,perm):
		N = self.N
		SIZE = self.SIZE
		pos2 = {perm[i]: (i % N, (i // N) % N, 0) for i in range(SIZE)}
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
				f = 1 # 縦横を優先する変数 0:たて,1:よこ
				while t != s:
					# print(t)
					next = (np.inf, -1)
					for j in range(SIZE):
						if (A[j][t] == 1) and (d[j]+1 == d[t]):
							if ((f == 0) and ((abs(perm_inv[j]-perm_inv[t]) == self.M*(SIZE/self.M-1)) or (abs(perm_inv[j]-perm_inv[t]) == self.M))) or ((f == 1) and ((abs(perm_inv[j]-perm_inv[t]) == 1) or (abs(perm_inv[j]-perm_inv[t]) == self.M-1))):
								next = min(next,(0,j))
							else:
								next = min(next,(1,j))
					if next[0] == np.inf:
						print("FAILED")
						sys.exit(1)
					prev = next[1]
					cost[t][prev] += C[s][i]
					cost[prev][t] += C[s][i]
					t = prev
					f = 1 - f

			return cost

		dis = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])
		for i in range(SIZE):
			dis += cal_cost(i)

		return dis


class Visualizer2D:
	def __init__(self,model,filename):
		self.model = model
		self.filename = filename
		self.red = -1
		self.green = -1
		self.sum = -1

	@staticmethod
	def curve_line(p0, p1, steps=20, amplitude=1.0):
		"""
		p0: 始点 (x0, y0, z0)
		p1: 終点 (x1, y1, z1)
		steps: 曲線の分割数
		amplitude: z軸への曲がりの強さ
		"""
		p0 = np.array(p0)
		p1 = np.array(p1)

		t = np.linspace(0, 1, steps)

		# 線形補間（x, y, z）
		x = (1 - t) * p0[0] + t * p1[0]
		y = (1 - t) * p0[1] + t * p1[1]
		z_linear = (1 - t) * p0[2] + t * p1[2]

		# z方向の膨らみ（sinカーブで左右対称に持ち上げる）
		z_bump = z_linear + amplitude * np.sin(np.pi * t)

		return x, y, z_bump


	def generate_graph(self,filename,score,perm,A,C):
		SIZE = self.model.SIZE
		
		G = nx.Graph()
		dis = self.model.calculate_cost(A,C,perm)
		count = np.count_nonzero(dis)
		if self.red == -1:
			self.sum = np.sum(dis)
			self.red = sorted(dis.ravel())[-int(0.3*count)]
			self.green = sorted(dis.ravel())[int(0.3*count+(SIZE*SIZE-count))]
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

		perm_inv=np.empty_like(perm)
		perm_inv[perm]=np.arange(len(perm))

		edge_traces=[]
		for u,v in G.edges():
			x=[pos[u][0],pos[v][0],None]
			y=[pos[u][1],pos[v][1],None]
			z=[pos[u][2],pos[v][2],None]

			if (abs(perm_inv[v]-perm_inv[u])==1) or (abs(perm_inv[v]-perm_inv[u])==math.sqrt(SIZE)): # N!=Mだとバグります(深刻)
				edge_traces.append(go.Scatter3d(
					x=x,y=y,z=z,
					mode='lines',
					line=dict(color=edgecolor[u][v],width=width[u][v])
				))
			elif (abs(perm_inv[v]-perm_inv[u])==math.sqrt(SIZE)-1):
				curve=self.curve_line(np.array(pos[u]),np.array(pos[v]))
				edge_traces.append(go.Scatter3d(
					x=curve[0],y=curve[1],z=curve[2],
					mode='lines',
					line=dict(color=edgecolor[u][v],width=width[u][v])
				))
			else:
				curve=self.curve_line(np.array(pos[u]),np.array(pos[v]),amplitude=-1.0)
				edge_traces.append(go.Scatter3d(
					x=curve[0],y=curve[1],z=curve[2],
					mode='lines',
					line=dict(color=edgecolor[u][v],width=width[u][v])
				))
		fig = go.Figure(data=[node_trace]+edge_traces)
		fig.update_layout(scene=dict(xaxis=dict(visible=False),
							yaxis=dict(visible=False),
							zaxis=dict(visible=False),aspectmode='data'))
		
		fig.write_html(f"{filename}.html")


if __name__ == "__main__":
	N,M = 5,5
	SIZE = N * M
	steps = 100000
	filename = "2torus"

	rng = np.random.default_rng()
	Rand = [[rng.integers(0,100).item() for _ in range(SIZE)] for _ in range(SIZE)]
	C = (np.tril(Rand) + np.tril(Rand).T - 2*np.diag(np.diag(Rand))).astype(np.int64) # 対角成分が0であるトラフィック量を表す対称行列C

	main(lambda:Torus2Dmodel(N,M,filename),SIZE,steps,filename,C)