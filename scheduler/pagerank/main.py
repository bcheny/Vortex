import numpy as np
import scipy.sparse as sp
import time

def load_edge_list(path, delimiter=None, dtype=np.int64, max_edges=None):
    edges = []
    nodes = set()
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_edges and i >= max_edges:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split() if delimiter is None else line.split(delimiter)
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except:
                continue
            edges.append((u, v))
            nodes.add(u); nodes.add(v)
    return edges, nodes

def build_sparse_transition(edges, node_to_idx, n_nodes, dtype=np.float32):
    outdeg = np.zeros(n_nodes, dtype=np.int64)
    rows = []
    cols = []
    for (u, v) in edges:
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        rows.append(iv)  # row index = destination
        cols.append(iu)  # col index = source
        outdeg[iu] += 1

    data = np.ones(len(rows), dtype=dtype)
    M = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=dtype)
    nonzero_cols = outdeg > 0
    inv_out = np.zeros(n_nodes, dtype=dtype)
    inv_out[nonzero_cols] = 1.0 / outdeg[nonzero_cols].astype(dtype)
    M = M.multiply(inv_out)
    return M, outdeg

def pagerank_power_iteration(M, outdeg, n_nodes, alpha=0.85, tol=1e-6, max_iter=100, dtype=np.float32, verbose=True):
    r = np.ones(n_nodes, dtype=dtype) / n_nodes
    teleport = (1.0 - alpha) / n_nodes
    dangling = (outdeg == 0)

    for it in range(1, max_iter + 1):
        t0 = time.time()
        dangling_sum = r[dangling].sum() if dangling.any() else 0.0

        # r_new = alpha * (M @ r) + alpha * dangling_sum / N + teleport
        Mr = M.dot(r)  # sparse matrix-vector multiply
        r_new = alpha * Mr
        if dangling_sum:
            r_new += alpha * (dangling_sum / n_nodes)
        r_new += teleport

        r_new /= r_new.sum()

        diff = np.linalg.norm(r_new - r, 1)
        if verbose:
            print(f"iter {it:3d}: L1 diff = {diff:.6e}, time = {time.time()-t0:.3f}s")
        r = r_new
        if diff < tol:
            if verbose:
                print(f"Converged after {it} iterations (L1 diff {diff:.2e} < tol {tol}).")
            break
    return r

def main(edge_list_path, dtype=np.float32, alpha=0.85, tol=1e-6, max_iter=100, verbose=True):
    print("Loading edges...")
    edges, nodes = load_edge_list(edge_list_path)
    print(f"Loaded edges: {len(edges):,}, unique nodes: {len(nodes):,}")

    sorted_nodes = sorted(nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    n_nodes = len(sorted_nodes)

    print("Building transition matrix...")
    M, outdeg = build_sparse_transition(edges, node_to_idx, n_nodes, dtype=dtype)
    print("Sparse matrix built.")
    print(f"M shape: {M.shape}, nnz: {M.nnz:,}, dtype: {M.dtype}")

    print("Running PageRank...")
    ranks = pagerank_power_iteration(M, outdeg, n_nodes, alpha=alpha, tol=tol, max_iter=max_iter, dtype=dtype, verbose=verbose)

    top_k = 20
    order = np.argsort(-ranks)[:top_k]
    print(f"\nTop {top_k} nodes by PageRank:")
    for rank_idx in order:
        print(f"node={idx_to_node[rank_idx]}, rank={ranks[rank_idx]:.6e}")

    out_file = "pagerank_scores.txt"
    with open(out_file, 'w') as fo:
        for i, score in enumerate(ranks):
            fo.write(f"{idx_to_node[i]}\t{score:.12e}\n")
    print(f"Saved pagerank scores to {out_file}")

if __name__ == "__main__":
    time.sleep(60)
    edge_list_path = 'web-Google.txt'
    main(edge_list_path)
