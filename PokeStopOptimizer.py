import cvxpy as cp
import numpy as np

# number of nodes
N = 16

# the time constraint is by default = 10 minutes (300 seconds)
Tmax = 300

# T = time it takes to travel from node i to node j, 
# Use x-large number if node is not connected in graph
T = np.full((N, N), 100000000)
T[0,1] = 20
T[0,2] = 30
T[0,7] = 90
T[1,0] = 20
T[1,2] = 40
T[1,3] = 50
T[1,6] = 70
T[2,0] = 30
T[2,1] = 40
T[2,7] = 70
T[3,1] = 50
T[3,4] = 10
T[3,5] = 30
T[3,6] = 50
T[4,3] = 10
T[4,5] = 30
T[4,11] = 100
T[4,12] = 110
T[4,13] = 130
T[5,3] = 30
T[5,4] = 30
T[5,6] = 30
T[5,8] = 40
T[5,9] = 50
T[5,10] = 70
T[5,11] = 80
T[5,12] = 100
T[6,1] = 70
T[6,3] = 50
T[6,5] = 30
T[6,7] = 10
T[6,8] = 10
T[7,0] = 90
T[7,2] = 70
T[7,6] = 10
T[7,8] = 10
T[8,5] = 40
T[8,6] = 10
T[8,7] = 10
T[8,9] = 10
T[8,11] = 80
T[9,5] = 50
T[9,8] = 10
T[9,10] = 20
T[9,11] = 60
T[10,5] = 70
T[10,9] = 20
T[10,11] = 30
T[11,4] = 100
T[11,5] = 80
T[11,8] = 80
T[11,9] = 60
T[11,10] = 30
T[11,12] = 30
T[11,14] = 60
T[12,4] = 110
T[12,5] = 100
T[12,11] = 30
T[12,13] = 40
T[12,14] = 20
T[12,15] = 50
T[13,4] = 130
T[13,12] = 40
T[13,14] = 50
T[13,15] = 20
T[14,11] = 60
T[14,12] = 50
T[14,15] = 30

# the scores for each node
S = np.full(N, 1)
S[2] = 2
S[12] = 2

# decision variables

# X = 1 if a visit to vertex i is followed by a visit to vertex j; 0 otherwise
# by setting X to only boolean values, equation 7 is met
X = cp.Variable((N, N), boolean  = True)

# U is the position of vertex i in the path
U = cp.Variable(N, integer = True)

# constraints
constraints = []
# equation 2
constraints.append(cp.sum(X[0,1:N]) == 1)
constraints.append(cp.sum(X[0:N-1,N-1]) == 1)

# equation 3
for k in range(1, N-1):
    constraints.append(cp.sum(X[0:N-1, k]) <= 1)
    constraints.append(cp.sum(X[k, 1:N]) <= 1)
    constraints.append(cp.sum(X[0:N-1, k]) == cp.sum(X[k, 1:N]))

# equation 4
constraints.append(cp.sum(cp.multiply(T[0:N-1, 1:N], X[0:N-1, 1:N])) <= Tmax)

# equation 5
for i in range(1, N):
  constraints.append(1 <= U[i])
  constraints.append(U[i] <= N-1)

# equation 6
for i in range(1, N):
  for j in range(1, N):
    constraints.append(U[i] - U[j] +  1 <= (N-1) * (1 - X[i, j]))

# objective function
obj_func = cp.sum(S[1:N-1] * X[1:N-1, 1:N])

problem = cp.Problem(cp.Maximize(obj_func), constraints)
problem.solve(solver=cp.GUROBI,verbose = False, qcp=True)

print("obj_func =")
print(obj_func.value)
print("U =")
print(U.value)
print("X =")
print(X.value)