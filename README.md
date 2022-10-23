# graph-code
Algorithms for coding paths on a graph.

Code structure:
1. Create/Load graphs into graph objects w. adjacency matrix from YAML/python files
2. Process graphs with algorithms:
	- lossless:
		- shortest path coding
	- lossy:
		- node coding
		- static path coding
		- dynamic path coding
	- lossy bounds:
		- distance bound
		- communication bound
		- distance + communication
3. Utility graph printing, solution printing

## Dependencies
1. python 3.10
2. pytorch
3. numpy
4. scipy
5. tqdm