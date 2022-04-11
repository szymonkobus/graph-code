# graph-code
Algorithms for coding paths on a graph.

Code structure:
1. Create/Load graphs into graph objects w. adjacency matrix from YAML/python files
2. Process graphs with algorithms:
	- lossless:
		- smallest complete subset (NP)
		- DAG (P)
	- lossy: (split equally, best split + rebalance, DAG -> binary, )
		- split equally
		- best split + rebalance
		- DAG -> binary
	- lossy bounds:
		- distance bound
		- communication bound
		- distance + communication
3. Utility graph printing, solution printing

## Dependencies
1. python 3.10
2. pytorch
3. numpy