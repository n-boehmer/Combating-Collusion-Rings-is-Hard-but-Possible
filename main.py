import gurobipy as gp
from gurobipy import GRB
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from itertools import product
import itertools
import networkx as nx
import collections
import os
import sys
import random
import tikzplotlib
import multiprocessing
from functools import partial
import os.path
import statistics

random.seed(0)
sys.setrecursionlimit(10000)
def simple_cycles(G, limit):
	subG = type(G)(G.edges())
	sccs = list(nx.strongly_connected_components(subG))
	while sccs:
		scc = sccs.pop()
		startnode = scc.pop()
		path = [startnode]
		blocked = set()
		blocked.add(startnode)
		stack = [(startnode, list(subG[startnode]))]

		while stack:
			thisnode, nbrs = stack[-1]

			if nbrs and len(path) < limit:
				nextnode = nbrs.pop()
				if nextnode == startnode:
					yield path[:]
				elif nextnode not in blocked:
					path.append(nextnode)
					stack.append((nextnode, list(subG[nextnode])))
					blocked.add(nextnode)
					continue
			if not nbrs or len(path) >= limit:
				blocked.remove(thisnode)
				stack.pop()
				path.pop()
		subG.remove_node(startnode)
		H = subG.subgraph(scc)
		sccs.extend(list(nx.strongly_connected_components(H)))
	return sccs

def solve_ILP(num_papers,num_reviewers,reviews_per_rev,reviews_per_paper,cycle_free,similarity_matrix,mask_matrix,egaliterian=False):
	rev_dict=[]
	for i in range(num_reviewers):
		rev_dict.append([j for j, x in enumerate(mask_matrix[i]) if x == 1])
	paper_dict=[]
	tr_mask_matrix=np.transpose(mask_matrix)
	for i in range(num_papers):
		paper_dict.append([j for j, x in enumerate(tr_mask_matrix[i]) if x == 1])
	m = gp.Model("mip1")
	m.setParam('OutputFlag', False)
	x = m.addVars(num_reviewers, num_papers, lb=0, ub=1, vtype=GRB.BINARY)
	opt = m.addVar(vtype=GRB.CONTINUOUS)

	for j in range(num_papers):
		m.addConstr(gp.quicksum(x[i, j] for i in range(num_reviewers)) == reviews_per_paper)

	for i in range(num_reviewers):
		m.addConstr(gp.quicksum(x[i, j] for j in range(num_papers)) <= reviews_per_rev)

	for j in range(num_papers):
		for i in range(num_reviewers):
			if mask_matrix[i, j] == 1:
				m.addConstr(x[i, j] == 0)
	if cycle_free>=2:
		for i in range(num_reviewers):
			for j in range(num_reviewers):
				for t in rev_dict[i]:
					for k in rev_dict[j]:
						m.addConstr(x[i, k] + x[j, t] <= 1)
	if cycle_free>=3:
		for i1 in range(num_reviewers):
			for i2 in range(num_reviewers):
				for i3 in range(num_reviewers):
					for p1 in rev_dict[i1]:
						for p2 in rev_dict[i2]:
							for p3 in rev_dict[i3]:
								m.addConstr(x[i1, p2] + x[i2, p3]+x[i3,p1]<= 2)
	if cycle_free >= 4:
		for i1 in range(num_reviewers):
			for i2 in range(num_reviewers):
				for i3 in range(num_reviewers):
					for i4 in range(num_reviewers):
						for p1 in rev_dict[i1]:
							for p2 in rev_dict[i2]:
								for p3 in rev_dict[i3]:
									for p4 in rev_dict[i4]:
										m.addConstr(x[i1, p2] + x[i2, p3] + x[i3, p4] + x[i4,p1] <= 3)
	if cycle_free >= 5:
		for i1 in range(num_reviewers):
			for i2 in range(num_reviewers):
				for i3 in range(num_reviewers):
					for i4 in range(num_reviewers):
						for i5 in range(num_reviewers):
							for p1 in rev_dict[i1]:
								for p2 in rev_dict[i2]:
									for p3 in rev_dict[i3]:
										for p4 in rev_dict[i4]:
											for p5 in rev_dict[i5]:
												m.addConstr(x[i1, p2] + x[i2, p3] + x[i3, p4] + x[i4,p5]+x[i5,p1]<= 4)
	if egaliterian:
		for j in range(num_papers):
			m.addConstr(gp.quicksum(similarity_matrix[i, j] * x[i, j] for i in range(num_reviewers)) >= opt)
	else:
		m.addConstr(
			gp.quicksum(similarity_matrix[i, j] * x[i, j] for i in range(num_reviewers) for j in range(num_papers)) == opt)
	m.setObjective(opt, GRB.MAXIMIZE)
	m.optimize()
	reviewer_graph =reviewerConflictGraph(x,paper_dict,num_reviewers,num_papers)
	paper_graph=paperConflictGraph(x,rev_dict,num_reviewers,num_papers)

	confs2=evaluateConflicts(reviewer_graph,paper_graph,2,num_reviewers,num_papers,x,rev_dict)
	confs3=evaluateConflicts(reviewer_graph,paper_graph,3,num_reviewers,num_papers,x,rev_dict)
	confs4=evaluateConflicts(reviewer_graph,paper_graph,4,num_reviewers,num_papers,x,rev_dict)
	confs5=evaluateConflicts(reviewer_graph,paper_graph,5,num_reviewers,num_papers,x,rev_dict)
	return m.objVal, confs2, confs3, confs4, confs5, collections.Counter([len(i) for i in nx.strongly_connected_components(reviewer_graph)]), collections.Counter([len(i) for i in nx.strongly_connected_components(paper_graph)])


def reviewerConflictGraph(x,paper_dict,num_reviewers,num_papers,ILP=True):
	edges=[]
	for i in range(num_reviewers):
		for j in range(num_papers):
			if ILP:
				if x[i,j].X==1:
					for t in paper_dict[j]:
						edges.append((i,t))
			else:
				if x[i,j]==1:
					for t in paper_dict[j]:
						edges.append((i,t))
	return nx.DiGraph(edges)

def paperConflictGraph(x,rev_dict,num_reviewers,num_papers,ILP=True):
	edges=[]
	for i in range(num_reviewers):
		for j in range(num_papers):
			if ILP:
				if x[i,j].X==1:
					for t in rev_dict[i]:
						edges.append((t,j))
			else:
				if x[i,j]==1:
					for t in rev_dict[i]:
						edges.append((t,j))
	return nx.DiGraph(edges)

def evaluateConflicts(reviewer_graph,paper_graph,length,num_reviewers,num_papers,x,rev_dict):
	reviewer_conflicts=list(simple_cycles(reviewer_graph, length+1))
	reviewer_conflicts_sorted = [sorted(agents) for agents in reviewer_conflicts]
	reviewer_conflicts = []
	for elem in reviewer_conflicts_sorted:
		if elem not in reviewer_conflicts:
			reviewer_conflicts.append(elem)
	paper_conflicts = list(simple_cycles(paper_graph, length+1))
	paper_conflicts_sorted = [sorted(papers) for papers in paper_conflicts ]
	paper_conflicts = []
	for elem in paper_conflicts_sorted:
		if elem not in paper_conflicts:
			paper_conflicts.append(elem)
	conflict_reviewer = set()
	conflict_papers = set()
	conflict_reviewer_dict = [0 for _ in range(num_reviewers)]
	conflict_paper_dict = [0 for _ in range(num_papers)]
	for rc in reviewer_conflicts:
		for r in rc:
			conflict_reviewer_dict[r] =conflict_reviewer_dict[r]+1
			conflict_reviewer.add(r)
	for pc in paper_conflicts:
		for p in pc:
			conflict_paper_dict[p] = conflict_paper_dict[p] + 1
			conflict_papers.add(p)
	r=0
	summed=0
	return [len(reviewer_conflicts),len(conflict_reviewer), len(conflict_papers),collections.Counter(conflict_reviewer_dict),collections.Counter(conflict_paper_dict),conflict_reviewer_dict[r],len(rev_dict[r]),summed]



def sample_conf(similarity_matrixG, mask_matrixG,num_papers,probability=1):
	sub_papers = sample(list(range(0, 911)), num_papers)
	sub_authors = set()
	for i in range(2435):
		for j in sub_papers:
			if mask_matrixG[i, j] == 1:
				sub_authors.add(i)
	sub_authorsprel = list(sub_authors)
	sub_authors=[]
	if probability<0:
		needed=int((-1)*num_papers*probability)+1
		sub_authors=sample(sub_authorsprel, needed)
	else:
		for i in sub_authorsprel:
			if random.uniform(0, 1) <= probability:
				sub_authors.append(i)
	similarity_matrix = similarity_matrixG[np.ix_(sub_authors, sub_papers)]
	mask_matrix = mask_matrixG[np.ix_(sub_authors, sub_papers)]

	return similarity_matrix, mask_matrix

def index_to_ag(id,num_papers):
	id_pap=id % num_papers
	return int((id-id_pap)/num_papers), id_pap

def greedy_cycle_free(num_papers,num_reviewers,reviews_per_rev,reviews_per_paper,similarity_matrix,mask_matrix,cf):
	list_similarity=similarity_matrix.flatten()
	rev_dict = []
	for i in range(num_reviewers):
		rev_dict.append([j for j, x in enumerate(mask_matrix[i]) if x == 1])
	paper_dict = []
	tr_mask_matrix = np.transpose(mask_matrix)
	for i in range(num_papers):
		paper_dict.append([j for j, x in enumerate(tr_mask_matrix[i]) if x == 1])
	sorted_indices=sorted(range(len(list_similarity)), key=lambda k: list_similarity[k])
	fixed_reviwes=np.zeros((num_reviewers,num_papers))
	rev_rev=np.zeros((num_reviewers,num_reviewers))
	rev_count=[0 for _ in range(num_reviewers)]
	paper_count = [0 for _ in range(num_papers)]
	for i in range(num_reviewers):
		for j in range(num_papers):
			if mask_matrix[i,j]==1:
				sorted_indices.remove(i*num_papers+j)
	done_papers=[]
	done_revs=[]
	not_done_papers=list(range(0,num_papers))
	not_done_revs = list(range(0, num_reviewers))
	DG = nx.DiGraph()
	DG_rev=nx.DiGraph()
	for i in range(num_papers):
		DG.add_node(i)
		DG_rev.add_node(i)
	for i in range(num_reviewers):
		DG.add_node(i+num_papers)
		DG_rev.add_node(i + num_papers)
	for i in range(num_papers):
		for ag in paper_dict[i]:
			DG.add_edge(i,ag+num_papers)
			DG_rev.add_edge(ag + num_papers,i)

	while len(done_papers)<num_papers and len(sorted_indices)>0 :
		cur_id=sorted_indices.pop()
		cur_rev_id,cur_pap_id = index_to_ag(cur_id,num_papers)
		if cur_rev_id not in done_revs and cur_pap_id not in done_papers:
			conflict=False
			if fixed_reviwes[cur_rev_id,cur_pap_id]==1:
				conflict=True
			if cur_rev_id+num_papers in nx.single_source_shortest_path_length(DG, cur_pap_id,cutoff=2*cf-1):
				conflict=True
			if not conflict:
				fixed_reviwes[cur_rev_id,cur_pap_id]=1
				for re in paper_dict[cur_pap_id]:
					rev_rev[cur_rev_id,re]=1
				rev_count[cur_rev_id]=rev_count[cur_rev_id]+1
				if rev_count[cur_rev_id]==reviews_per_rev:
					done_revs.append(cur_rev_id)
					not_done_revs.remove(cur_rev_id)
				paper_count[cur_pap_id]=paper_count[cur_pap_id]+1
				if paper_count[cur_pap_id]==reviews_per_paper:
					done_papers.append(cur_pap_id)
					not_done_papers.remove(cur_pap_id)
				DG.add_edge(cur_rev_id+num_papers, cur_pap_id)
				DG_rev.add_edge(cur_pap_id,cur_rev_id + num_papers)
	reviewer_graph = reviewerConflictGraph(fixed_reviwes, paper_dict, num_reviewers, num_papers, ILP=False)
	paper_graph = paperConflictGraph(fixed_reviwes, rev_dict, num_reviewers, num_papers, ILP=False)

	confs2 = evaluateConflicts(reviewer_graph, paper_graph, 2, num_reviewers, num_papers, fixed_reviwes, rev_dict)
	confs3 = evaluateConflicts(reviewer_graph, paper_graph, 3, num_reviewers, num_papers, fixed_reviwes, rev_dict)
	while len(done_papers) < num_papers:
		cur_pap_id=not_done_papers[random.randint(0, len(not_done_papers)-1)]
		cur_rev_id=not_done_revs[random.randint(0, len(not_done_revs)-1)]
		possible_paper_to_review=[t for t in range(num_papers) if t not in nx.single_source_shortest_path_length(DG_rev, cur_rev_id+num_papers,cutoff=2*cf-1) and not fixed_reviwes[cur_rev_id,t]==1]
		possible_agents_to_review=[t for t in range(num_reviewers) if t+num_papers not in nx.single_source_shortest_path_length(DG, cur_pap_id,cutoff=2*cf-1) and not fixed_reviwes[t,cur_pap_id]==1]
		bpair=[]
		best=-9999
		for i in possible_agents_to_review:
			for j in possible_paper_to_review:
				if fixed_reviwes[i,j]==1:
					if similarity_matrix[cur_rev_id,j]+similarity_matrix[i,cur_pap_id]-similarity_matrix[i,j]>best:
						best=similarity_matrix[cur_rev_id,j]+similarity_matrix[i,cur_pap_id]-similarity_matrix[i,j]
						bpair=[i,j]
		try:
			fixed_reviwes[bpair[0], bpair[1]] = 0
		except:
			print('Recursive Call')
			return greedy_cycle_free(num_papers,num_reviewers,reviews_per_rev,reviews_per_paper,similarity_matrix,mask_matrix,cf)
		fixed_reviwes[bpair[0], bpair[1]] = 0
		DG.remove_edge(bpair[0] + num_papers, bpair[1])
		DG_rev.remove_edge(bpair[1],bpair[0] + num_papers)
		fixed_reviwes[cur_rev_id, bpair[1]]=1
		DG.add_edge(cur_rev_id+ num_papers, bpair[1])
		DG_rev.add_edge(bpair[1],cur_rev_id + num_papers)
		fixed_reviwes[bpair[0], cur_pap_id] = 1
		DG.add_edge(bpair[0] + num_papers, cur_pap_id)
		DG_rev.add_edge(cur_pap_id,bpair[0] + num_papers)
		rev_count[cur_rev_id] = rev_count[cur_rev_id] + 1
		if rev_count[cur_rev_id] == reviews_per_rev:
			done_revs.append(cur_rev_id)
			not_done_revs.remove(cur_rev_id)
		paper_count[cur_pap_id] = paper_count[cur_pap_id] + 1
		if paper_count[cur_pap_id] == reviews_per_paper:
			done_papers.append(cur_pap_id)
			not_done_papers.remove(cur_pap_id)
	summed=0
	for i in range(num_reviewers):
		for j in range(num_papers):
			if fixed_reviwes[i,j]==1:
				summed+=similarity_matrix[i,j]


	reviewer_graph =reviewerConflictGraph(fixed_reviwes,paper_dict,num_reviewers,num_papers,ILP=False)
	paper_graph=paperConflictGraph(fixed_reviwes,rev_dict,num_reviewers,num_papers,ILP=False)

	confs2=evaluateConflicts(reviewer_graph,paper_graph,2,num_reviewers,num_papers,fixed_reviwes,rev_dict)
	confs3=evaluateConflicts(reviewer_graph,paper_graph,3,num_reviewers,num_papers,fixed_reviwes,rev_dict)
	confs4=evaluateConflicts(reviewer_graph,paper_graph,4,num_reviewers,num_papers,fixed_reviwes,rev_dict)
	confs5=evaluateConflicts(reviewer_graph,paper_graph,5,num_reviewers,num_papers,fixed_reviwes,rev_dict)

	return summed, confs2, confs3, confs4,confs5

def center_method_proba(similarity_matrixG,mask_matrixG,num_iterations,lengthfree,num_revs_per_rev,num_revs_per_pa,secondILP,skipgreedy,s,probability):
	n_conflicts = [0, 0, 0, 0]
	n_revs = [0, 0, 0, 0]
	n_papers = [0, 0, 0, 0]
	quality = 0

	quality_cf = [0 for i in range(0, len(lengthfree))]
	quality_fraction = [0 for i in range(0, len(lengthfree))]
	cf_n_conflicts = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	cf_n_revs = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	cf_n_papers = [[0, 0, 0, 0] for _ in range(len(lengthfree))]

	quality_grcf = [0 for i in range(0, len(lengthfree))]
	quality_fractiongr = [0 for i in range(0, len(lengthfree))]
	grcf_n_conflicts = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	grcf_n_revs = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	grcf_n_papers = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	for i in range(num_iterations):
		similarity_matrix, mask_matrix = sample_conf(similarity_matrixG, mask_matrixG, s, probability)
		num_pap = similarity_matrix.shape[1]
		num_ag = similarity_matrix.shape[0]
		print(num_ag)
		a, b, c, d, e, f, g = solve_ILP(num_pap, num_ag, num_revs_per_rev, num_revs_per_pa, 0, similarity_matrix,
										mask_matrix)
		quality = quality + a/num_iterations
		for counter, value in enumerate([b, c, d, e]):
			n_conflicts[counter] = n_conflicts[counter] + value[0] / num_iterations
			n_revs[counter] = n_revs[counter] + (value[1] / (num_ag * num_iterations))
			n_papers[counter] = n_papers[counter] + (value[2] / (num_pap * num_iterations))
		for counter, value in enumerate(lengthfree):
			if value in secondILP:
				a2, b2, c2, d2, e2, f2, g2= solve_ILP(num_pap, num_ag, num_revs_per_rev, num_revs_per_pa, value,
													   similarity_matrix, mask_matrix)
				quality_cf[counter] = quality_cf[counter] + a2 / (num_iterations)
				quality_fraction[counter] = quality_fraction[counter] + (a2 / (a * num_iterations))
				for counter2, value2 in enumerate([b2, c2, d2, e2]):
					cf_n_conflicts[counter][counter2] = cf_n_conflicts[counter][counter2] + value2[0] / num_iterations
					cf_n_revs[counter][counter2] = cf_n_revs[counter][counter2] + (
								value2[1] / (num_ag * num_iterations))
					cf_n_papers[counter][counter2] = cf_n_papers[counter][counter2] + (
								value2[2] / (num_pap * num_iterations))
			if not skipgreedy:
				a2, b2, c2, d2, e2 = greedy_cycle_free(num_pap, num_ag, num_revs_per_rev, num_revs_per_pa,
													   similarity_matrix, mask_matrix, value)
				quality_grcf[counter] = quality_grcf[counter] + a2 / (num_iterations)
				quality_fractiongr[counter] = quality_fractiongr[counter] + (a2 / (a * num_iterations))
				for counter2, value2 in enumerate([b2, c2, d2, e2]):
					grcf_n_conflicts[counter][counter2] = grcf_n_conflicts[counter][counter2] + value2[0] / num_iterations
					grcf_n_revs[counter][counter2] = grcf_n_revs[counter][counter2] + (
							value2[1] / (num_ag * num_iterations))
					grcf_n_papers[counter][counter2] = grcf_n_papers[counter][counter2] + (
							value2[2] / (num_pap * num_iterations))
	return n_revs, n_papers,  quality_fraction,  cf_n_revs, cf_n_papers,  quality_fractiongr, grcf_n_revs, grcf_n_papers

def center_method(similarity_matrixG,mask_matrixG,num_iterations,lengthfree,num_revs_per_rev,num_revs_per_pa,probability,secondILP,skipgreedy,s):
	n_conflicts = [0, 0, 0, 0]
	n_revs = [0, 0, 0, 0]
	n_papers = [0, 0, 0, 0]
	quality = 0
	quality_cf = [0 for i in range(0, len(lengthfree))]
	quality_fraction = [0 for i in range(0, len(lengthfree))]
	cf_n_conflicts = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	cf_n_revs = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	cf_n_papers = [[0, 0, 0, 0] for _ in range(len(lengthfree))]

	quality_grcf = [0 for i in range(0, len(lengthfree))]
	quality_fractiongr = [0 for i in range(0, len(lengthfree))]
	grcf_n_conflicts = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	grcf_n_revs = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	grcf_n_papers = [[0, 0, 0, 0] for _ in range(len(lengthfree))]
	for i in range(num_iterations):
		similarity_matrix, mask_matrix = sample_conf(similarity_matrixG, mask_matrixG, s, probability)
		num_pap = similarity_matrix.shape[1]
		num_ag = similarity_matrix.shape[0]
		print(num_pap)
		a, b, c, d, e, f, g = solve_ILP(num_pap, num_ag, num_revs_per_rev, num_revs_per_pa, 0, similarity_matrix,
										mask_matrix)
		quality = quality + a/num_iterations
		for counter, value in enumerate([b, c, d, e]):
			n_conflicts[counter] = n_conflicts[counter] + value[0] / num_iterations
			n_revs[counter] = n_revs[counter] + (value[1] / (num_ag * num_iterations))
			n_papers[counter] = n_papers[counter] + (value[2] / (num_pap * num_iterations))
		for counter, value in enumerate(lengthfree):
			if value in secondILP:
				a2, b2, c2, d2, e2, f2, g2 = solve_ILP(num_pap, num_ag, num_revs_per_rev, num_revs_per_pa, value,
													   similarity_matrix, mask_matrix)
				quality_cf[counter] = quality_cf[counter] + a2 / (num_iterations)
				quality_fraction[counter] = quality_fraction[counter] + (a2 / (a * num_iterations))
				for counter2, value2 in enumerate([b2, c2, d2, e2]):
					cf_n_conflicts[counter][counter2] = cf_n_conflicts[counter][counter2] + value2[0] / num_iterations
					cf_n_revs[counter][counter2] = cf_n_revs[counter][counter2] + (
								value2[1] / (num_ag * num_iterations))
					cf_n_papers[counter][counter2] = cf_n_papers[counter][counter2] + (
								value2[2] / (num_pap * num_iterations))
			if not skipgreedy:
				a2, b2, c2, d2, e2 = greedy_cycle_free(num_pap, num_ag, num_revs_per_rev, num_revs_per_pa,
													   similarity_matrix, mask_matrix, value)
				quality_grcf[counter] = quality_grcf[counter] + a2 / (num_iterations)
				quality_fractiongr[counter] = quality_fractiongr[counter] + (a2 / (a * num_iterations))
				for counter2, value2 in enumerate([b2, c2, d2, e2]):
					grcf_n_conflicts[counter][counter2] = grcf_n_conflicts[counter][counter2] + value2[0] / num_iterations
					grcf_n_revs[counter][counter2] = grcf_n_revs[counter][counter2] + (
							value2[1] / (num_ag * num_iterations))
					grcf_n_papers[counter][counter2] = grcf_n_papers[counter][counter2] + (
							value2[2] / (num_pap * num_iterations))
	return n_revs, n_papers,  quality_fraction,  cf_n_revs, cf_n_papers,  quality_fractiongr, grcf_n_revs, grcf_n_papers



def par_experiments_size(similarity_matrixG,mask_matrixG,num_iterations,sizes,lengthfree,num_revs_per_rev,num_revs_per_pa,probability,secondILP=[],skipgreedy=False, name=''):
	"""Method to execute our first experiments.

	    Keyword arguments:
	    similarity_matrixG -- unfiltered similarity matrix of agents and papers
	    mask_matrix& -- unifiltered mask matrix indicating which agent wrote which paper
	    num_iterations -- number of sampled instances
	    sizes -- list of different numbers of papers for which experiment is executed
	    lengthfree -- list of different cycle lengths we want to exclude
	    num_revs_per_rev -- maximum number of reviews each agent can write
	    num_revs_per_pa -- number of reviews per paper needed
	    probability -- (-1)*number of agents / number of papers
	    secondILP -- list of different lengths l for which optimal l-cycle-free solution is computed
	    skipgreedy -- bool indicating whether the execution of our heuristic is skipped
	    name -- name of our experiment

	"""
	colors = ['darkred', 'darkorange', 'forestgreen', 'dodgerblue','orchid']
	folder="p"+str(num_revs_per_pa)+"_r_"+str(num_revs_per_rev)
	if not os.path.exists("./"+folder):
		os.makedirs("./"+folder)
	pool = multiprocessing.Pool(processes=os.cpu_count())
	partial_center = partial(center_method, similarity_matrixG, mask_matrixG, num_iterations, lengthfree, num_revs_per_rev, num_revs_per_pa,
				  probability, secondILP,skipgreedy)
	ret=pool.map(partial_center, sizes)
	l_n_revs = []
	l_n_papers = []
	l_quality_frac = []
	l_cf_n_revs = []
	l_cf_n_papers = []
	l_quality_fracgr = []
	l_cf_n_revsgr = []
	l_cf_n_papersgr = []
	for r in ret:
		l_n_revs.append(r[0])
		l_n_papers.append(r[1])
		l_quality_frac.append(r[2])
		l_cf_n_revs.append(r[3])
		l_cf_n_papers.append(r[4])
		l_quality_fracgr.append(r[5])
		l_cf_n_revsgr.append(r[6])
		l_cf_n_papersgr.append(r[7])

	for i in range(0, 3):
		plt.plot(sizes, [l_n_revs[j][i] for j in range(len(sizes))],label="in length"+str(i+2)+ " cycle in optimal",color=colors[i],linestyle='solid')
	for k in range(len(lengthfree)):
		for i in range(0, 3):
			plt.plot(sizes, [l_cf_n_revs[j][k][i] for j in range(len(sizes))], label="in length" + str(i + 2)+ " cycle in optimal cf  "+str(lengthfree[k]),color=colors[i],linestyle='dashed')
			plt.plot(sizes, [l_cf_n_revsgr[j][k][i] for j in range(len(sizes))],
					 label=" in length" + str(i + 2) + " cycle in heuristic cf " + str(lengthfree[k]),color=colors[i],linestyle='dotted')
	plt.legend()
	plt.xlabel('num_papers')
	plt.ylabel('fraction reviwers in rev cylce')
	plt.savefig("./"+folder+'/frac_revs'+name+'.png')
	tikzplotlib.save("./"+folder+'/frac_revs'+name+'.tex', encoding='utf-8')
	plt.close()
	for i in range(0, 3):
		plt.plot(sizes, [l_n_papers[j][i] for j in range(len(sizes))], label="in length" + str(i + 2)+ " cycle in optimal",color=colors[i],linestyle='solid')
	for k in range(len(lengthfree)):
		for i in range(0, 3):
			plt.plot(sizes, [l_cf_n_papers[j][k][i] for j in range(len(sizes))], label="in length" + str(i + 2)+ " cycle in optimal cf "+str(lengthfree[k]),color=colors[i],linestyle='dashed')
			plt.plot(sizes, [l_cf_n_papersgr[j][k][i] for j in range(len(sizes))],
					 label="in length" + str(i + 2) + " cycle in heuristic cf " + str(lengthfree[k]),color=colors[i],linestyle='dotted')
	plt.legend()
	plt.xlabel('num_papers')
	plt.ylabel('fraction paper in rev cylce')
	plt.savefig("./"+folder+'/frac_paper'+name+'.png')
	tikzplotlib.save("./"+folder+'/frac_paper'+name+'.tex', encoding='utf-8')
	plt.close()
	if not skipgreedy:
		for counter, value in enumerate(lengthfree):
			plt.plot(sizes, [l_quality_fracgr[j][counter] for j in range(len(sizes))], label="heuristic cf length" + str(value),color=colors[counter],linestyle='dotted')
	for counter, value in enumerate(secondILP):
		plt.plot(sizes, [l_quality_frac[j][counter] for j in range(len(sizes))], label="optimal cf length" + str(value),
				 color=colors[counter], linestyle='dashed')

	plt.legend()
	plt.xlabel('num_papers')
	plt.ylabel('fraction quality decrease')
	plt.savefig("./"+folder+'/quality'+name+'.png')
	tikzplotlib.save("./"+folder+'/quality'+name+'.tex', encoding='utf-8')
	plt.close()

def par_experiments_size_proba(similarity_matrixG,mask_matrixG,num_iterations,size,lengthfree,num_revs_per_rev,num_revs_per_pa,probabilities,secondILP=[],skipgreedy=False, name=''):
	colors = ['darkred', 'darkorange', 'forestgreen', 'dodgerblue','orchid']
	folder="p"+str(num_revs_per_pa)+"_r_"+str(num_revs_per_rev)
	if not os.path.exists("./"+folder):
		os.makedirs("./"+folder)
	pool = multiprocessing.Pool(processes=os.cpu_count())
	partial_center = partial(center_method_proba, similarity_matrixG, mask_matrixG, num_iterations, lengthfree, num_revs_per_rev, num_revs_per_pa,
				  secondILP,skipgreedy,size)
	ret=pool.map(partial_center, probabilities)
	l_n_revs = []
	l_n_papers = []
	l_quality_frac = []
	l_cf_n_revs = []
	l_cf_n_papers = []
	l_quality_fracgr = []
	l_cf_n_revsgr = []
	l_cf_n_papersgr = []
	for r in ret:
		l_n_revs.append(r[0])
		l_n_papers.append(r[1])
		l_quality_frac.append(r[2])
		l_cf_n_revs.append(r[3])
		l_cf_n_papers.append(r[4])
		l_quality_fracgr.append(r[5])
		l_cf_n_revsgr.append(r[6])
		l_cf_n_papersgr.append(r[7])

	pros=[(-1)*p for p in probabilities]
	for i in range(0, 3):
		plt.plot(pros, [l_n_revs[j][i] for j in range(len(pros))],label="in length"+str(i+2)+ " in optimal",color=colors[i],linestyle='solid')
	for k in range(len(lengthfree)):
		for i in range(0, 3):
			plt.plot(pros, [l_cf_n_revs[j][k][i] for j in range(len(pros))], label="in length" + str(i + 2)+ " in optimal cf "+str(lengthfree[k]),color=colors[i],linestyle='dashed')
			plt.plot(pros, [l_cf_n_revsgr[j][k][i] for j in range(len(pros))],
					 label="in length" + str(i + 2) + " in heuristic cf " + str(lengthfree[k]),color=colors[i],linestyle='dotted')
	plt.legend()
	plt.xlabel('num_papers')
	plt.ylabel('fraction rev in rev cylce')
	plt.savefig("./"+folder+'/frac_revs'+name+'.png')
	tikzplotlib.save("./"+folder+'/frac_revs'+name+'.tex', encoding='utf-8')
	plt.close()
	for i in range(0, 3):
		plt.plot(pros, [l_n_papers[j][i] for j in range(len(pros))], label="in length" + str(i + 2)+ " in optimal",color=colors[i],linestyle='solid')
	for k in range(len(lengthfree)):
		for i in range(0, 3):
			plt.plot(pros, [l_cf_n_papers[j][k][i] for j in range(len(pros))], label="in length" + str(i + 2)+ " in optimal cf "+str(lengthfree[k]),color=colors[i],linestyle='dashed')
			plt.plot(pros, [l_cf_n_papersgr[j][k][i] for j in range(len(pros))],
					 label="in length" + str(i + 2) + " in heuristic cf " + str(lengthfree[k]),color=colors[i],linestyle='dotted')
	plt.legend()
	plt.xlabel('num_papers')
	plt.ylabel('fraction paper in rev cylce')
	plt.savefig("./"+folder+'/frac_paper'+name+'.png')
	tikzplotlib.save("./"+folder+'/frac_paper'+name+'.tex', encoding='utf-8')
	plt.close()
	if not skipgreedy:
		for counter, value in enumerate(lengthfree):
			plt.plot(pros, [l_quality_fracgr[j][counter] for j in range(len(pros))], label="length heuristic cf" + str(value),color=colors[counter],linestyle='dotted')
	for counter, value in enumerate(secondILP):
		plt.plot(pros, [l_quality_frac[j][counter] for j in range(len(pros))], label="length optimal cf" + str(value),
				 color=colors[counter], linestyle='dashed')
	plt.legend()
	plt.xlabel('num_papers')
	plt.ylabel('fraction quality decrease')
	plt.savefig("./"+folder+'/quality'+name+'.png')
	tikzplotlib.save("./"+folder+'/quality'+name+'.tex', encoding='utf-8')
	plt.close()


scores = np.load("iclr2018.npz", allow_pickle=True)
similarity_matrixG = scores["similarity_matrix"]
mask_matrixG = scores["mask_matrix"]


nums=200

#Experiment I
random.seed(0)
par_experiments_size(similarity_matrixG,mask_matrixG,nums,list(range(150, 901, 25)),[2,3,4],6,3,probability=-0.5,secondILP=[2],name='expI')
#We seperately generate the results for optimum 3-cycle-free
#WARNING you need around 20GB RAM to run the following line
random.seed(0)
#par_experiments_size(similarity_matrixG,mask_matrixG,nums,list(range(150, 226, 25)),[3],6,3,probability=-0.5,secondILP=[3],skipgreedy=True,name='expI_cf3')

#Exeperiment II
random.seed(0)
#WARNING you need around 40GB RAM to run the following line if you want to compute the results for the 3-cycle-free variant; if not, then set secondILP=[2]
#par_experiments_size_proba(similarity_matrixG,mask_matrixG,nums,150,[2,3,4],6,3,probabilities=[-0.5,-0.6,-0.7,-0.8,-0.9,-1,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7,-1.8,-1.9,-2],skipgreedy=False,secondILP=[2,3],name='expII')
