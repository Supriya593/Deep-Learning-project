from pulp import LpMinimize, LpProblem, LpVariable, lpSum

# Warehouses and Customers
warehouses = ['W1', 'W2']
customers = ['A', 'B', 'C']

# Supply and Demand
supply = {'W1': 100, 'W2': 150}
demand = {'A': 80, 'B': 120, 'C': 50}

# Shipping cost per unit
costs = {
    ('W1', 'A'): 2, ('W1', 'B'): 4, ('W1', 'C'): 5,
    ('W2', 'A'): 3, ('W2', 'B'): 1, ('W2', 'C'): 7
}

# Create problem
model = LpProblem("Transportation-Min-Cost", LpMinimize)

# Create variables
x = LpVariable.dicts("Ship", (warehouses, customers), lowBound=0, cat='Continuous')

# Objective function
model += lpSum(x[w][c] * costs[(w, c)] for w in warehouses for c in customers)

# Supply constraints
for w in warehouses:
    model += lpSum(x[w][c] for c in customers) <= supply[w]

# Demand constraints
for c in customers:
    model += lpSum(x[w][c] for w in warehouses) >= demand[c]

# Solve
model.solve()

# Results
print("Shipping Plan:")
for w in warehouses:
    for c in customers:
        print(f"{w} to {c}: {x[w][c].value()} units")

print(f"\nTotal Minimum Shipping Cost: ₹{model.objective.value()}")
