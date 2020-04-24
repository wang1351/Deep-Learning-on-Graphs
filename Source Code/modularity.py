import networkx as nx
import torch
from networkx.algorithms.community.quality import modularity
from networkx.utils.mapped_queue import MappedQueue
import heapq

def make_modularity_matrix(adj):
    size = adj.shape[1]
    adj = adj*(torch.ones(size, size) - torch.eye(size))
    degrees = adj.sum(0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

def greedy_modularity_communities(G, K):
    
#  Code modified from https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/community/modularity_max.html#greedy_modularity_communities

    G = nx.from_numpy_array(G.detach().numpy(), nx.Graph())
    node_num = len(G.nodes())
    m = sum([d.get('weight', 1) for u, v, d in G.edges(data=True)])
    q0 = 1.0 / (2.0 * m)
    label_for_node = {}
    for i, v in enumerate(G.nodes()):
        label_for_node[i] = v
    node_for_label = {}
    communities = {}
    k = []
    a = []
    merges = []
    degree_for_label = G.degree(G.nodes())
    for i in range(node_num):
        node_for_label[label_for_node[i]] = i
        communities[i] = frozenset([i])
        k.append(degree_for_label[label_for_node[i]])
        a.append(q0*k[i])
    partition = [[label_for_node[x] for x in c] for c in communities.values()]
    q_cnm = modularity(G, partition)

    dq_dict = dict(
        (i, dict(
            (j, q0 * (G[i][j]['weight'] + G[j][i]['weight']) - 2 * k[i] * k[j] * q0 * q0)
            for j in [
                node_for_label[u]
                for u in G.neighbors(label_for_node[i])]
            if j != i))
        for i in range(node_num))
    dq_heap = [MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(node_num)]
    H = MappedQueue([dq_heap[i].h[0] for i in range(node_num) if len(dq_heap[i]) > 0])
    # Merge communities until we can't improve modularity
    while len(H) > 1:
        dq, i, j = H.pop()
        dq = -dq
        dq_heap[i].pop()
        if len(dq_heap[i]) != 0:
            H.push(dq_heap[i].h[0])
        if dq_heap[j].h[0] == (-dq, j, i):
            H.remove((-dq, j, i))
            dq_heap[j].remove((-dq, j, i))
            if len(dq_heap[j]) > 0:
                H.push(dq_heap[j].h[0])
        else:
            dq_heap[j].remove((-dq, j, i))

        communities[j] = frozenset(communities[i] | communities[j])
        del communities[i]
        merges.append((i, j, dq))

        if len(communities) == K:
            break
        q_cnm += dq
        i_set = set(dq_dict[i].keys())
        j_set = set(dq_dict[j].keys())
        all_set = (i_set | j_set) - set([i, j])
        both_set = i_set & j_set
        for k in all_set:
            if k in both_set:
                dq_jk = dq_dict[j][k] + dq_dict[i][k]
            elif k in j_set:
                dq_jk = dq_dict[j][k] - 2.0 * a[i] * a[k]
            else:
                dq_jk = dq_dict[i][k] - 2.0 * a[j] * a[k]
            for row, col in [(j, k), (k, j)]:
                if k in j_set:
                    d_old = (-dq_dict[row][col], row, col)
                else:
                    d_old = None
                dq_dict[row][col] = dq_jk
                if len(dq_heap[row]) > 0:
                    d_oldmax = dq_heap[row].h[0]
                else:
                    d_oldmax = None
                d = (-dq_jk, row, col)
                if d_old is None:
                    dq_heap[row].push(d)
                else:
                    dq_heap[row].update(d_old, d)
                if d_oldmax is None:
                    H.push(d)
                else:
                    if dq_heap[row].h[0] != d_oldmax:
                        H.update(d_oldmax, dq_heap[row].h[0])
        i_neighbors = dq_dict[i].keys()
        for k in i_neighbors:
            dq_old = dq_dict[k][i]
            del dq_dict[k][i]
            if k != j:
                for row, col in [(k, i), (i, k)]:
                    d_old = (-dq_old, row, col)
                    if dq_heap[row].h[0] == d_old:
                        dq_heap[row].remove(d_old)
                        H.remove(d_old)
                        if len(dq_heap[row]) > 0:
                            H.push(dq_heap[row].h[0])
                    else:
                        dq_heap[row].remove(d_old)

        del dq_dict[i]
        dq_heap[i] = MappedQueue()
        a[j] += a[i]
        a[i] = 0
    heap = []
    for j in communities:
        heapq.heappush(heap, (a[j], set(communities[j])))
    while len(heap) > K:
        weight1, com1 = heapq.heappop(heap)
        weight2, com2 = heapq.heappop(heap)
        com1.update(com2)
        heapq.heappush(heap, (weight1 + weight2, com1))
    communities = [x[1] for x in heap]
    r = torch.zeros(node_num, K)
    for i, c in enumerate(communities):
        for v in c:
            r[v, i] = 1
    return r