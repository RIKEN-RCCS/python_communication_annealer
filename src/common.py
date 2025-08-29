# 総和の最小化
import numpy as np
from scipy import linalg
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from simanneal import Annealer
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import sys
from abc import ABC,abstractmethod
import os


rng = np.random.default_rng()
process=[]


class Graphmodel(ABC):
	def __init__(self,size):
		self.size = size

	@abstractmethod
	def generate_A(self):
		pass

	@abstractmethod
	def generate_pos(self,G,perm):
		pass

	@abstractmethod
	def calculate_cost(self,A,C):
		pass
		

@dataclass
class State:
	C: np.array
	D: np.array
	perm: np.array
	SIZE: int


class ProcessMapping:
	def __init__(self,model,A,D,C,SIZE,filename):
		self.model = model
		self.A = A
		self.D = D
		self.C = C
		self.SIZE = SIZE
		self.filename = filename

		self.red = -1
		self.green = -1
		self.sum = -1

	@staticmethod
	def generate_process_graph(filename):
		plt.figure()
		plt.suptitle("cost graph")
		plt.ylabel("cost")
		plt.xlabel("steps")
		x = list(range(len(process)))
		y = process

		plt.plot(x,y)

		plt.savefig(filename)
		plt.close()
		# os.system(f"img2sixel {filename}")

	def generate_C_graph(self,filename):
		adj_matrix = self.C

		num_nodes = adj_matrix.shape[0]

		plt.figure(figsize=(num_nodes, num_nodes))
		im = plt.imshow(adj_matrix, cmap='Blues', interpolation='none')

		# 軸ラベル
		plt.title("Traffic Matrix C",fontsize=self.SIZE*4)
		plt.xlabel("Node",fontsize=self.SIZE*3)
		plt.ylabel("Node",fontsize=self.SIZE*3)
		step = 10
		ticks = list(range(0, num_nodes, step))
		labels = [str(i) for i in ticks]

		plt.xticks(ticks=ticks, labels=labels,fontsize=self.SIZE*2)
		plt.yticks(ticks=ticks, labels=labels,fontsize=self.SIZE*2)

		
		plt.grid(False)
		plt.tight_layout()
		plt.savefig(filename)
		plt.close()

	def solve(self,steps=10000):
		SIZE = self.SIZE
		perm = np.arange(self.SIZE)

		initial_state = State(C=self.C, D=self.D, perm=perm, SIZE=SIZE)
		prob = Mapping(initial_state)
		first_score = prob.energy()
		prob.steps = steps
		prob.copy_strategy = "method" # これはなに
		state,final_score = prob.anneal()

		self.model.visualizer.generate_graph(f"../out/{self.filename}_1",first_score,perm,self.A,self.C)
		
		A_perm = np.array([[0 for _ in range(SIZE)] for _ in range(SIZE)])
		for i in range(SIZE):
			for j in range(SIZE):
				A_perm[state.perm[i]][state.perm[j]] = self.A[i][j]
		
		self.model.visualizer.generate_graph(f"../out/{self.filename}_2",final_score,state.perm,A_perm,self.C)
		self.generate_process_graph(f"../out/{self.filename}_p.png")
		self.generate_C_graph(f"../out/{self.filename}_c.png")
		# print(steps,final_score)


class Mapping(Annealer):
	def __init__(self,state):
		super(Mapping,self).__init__(state) # 継承

	# 移動を指定する関数
	def move(self):
		global process
		process.append(self.energy())
		# permを置換
		t1 = rng.integers(0,self.state.SIZE).item()
		t2 = rng.integers(0,self.state.SIZE).item()
		self.state.perm[t1],self.state.perm[t2] = self.state.perm[t2],self.state.perm[t1]

	# 良いスコア設定する関数
	def energy(self):
		C = self.state.C
		D = self.state.D
		R = np.eye(self.state.SIZE)[self.state.perm]
		cost = np.trace(C @ R.T @ D @ R)
		return cost

	def copy_state(self,state):
		return State(
			C = state.C,
			D = state.D,
			perm = np.copy(state.perm),
			SIZE = state.SIZE
		)


def main(model_class,SIZE,steps,filename,C):
	model = model_class()
	A = model.generate_A()
	D = shortest_path(csr_matrix(A)).astype(int)
		
	mapping = ProcessMapping(model,A,D,C,SIZE,filename)
	mapping.solve(steps)
	