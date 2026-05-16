"""
Causal DAG for Atlatl.
Pre-specified based on game code analysis.
Supports conditional independence queries for kernel design.
"""

from collections import defaultdict
import config


class CausalDAG:
    """Directed Acyclic Graph encoding causal structure."""

    def __init__(self, edges=None):
        self.parents = defaultdict(set)   # child -> set of parents
        self.children = defaultdict(set)  # parent -> set of children
        self.nodes = set()
        if edges is None:
            edges = config.CAUSAL_EDGES
        for parent, child, _ in edges:
            self.add_edge(parent, child)

    def add_edge(self, parent, child):
        self.parents[child].add(parent)
        self.children[parent].add(child)
        self.nodes.add(parent)
        self.nodes.add(child)

    def ancestors(self, node):
        """All ancestors of a node (recursive parents)."""
        visited = set()
        stack = list(self.parents[node])
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.parents[n])
        return visited

    def descendants(self, node):
        """All descendants of a node (recursive children)."""
        visited = set()
        stack = list(self.children[node])
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                stack.extend(self.children[n])
        return visited

    def get_causal_path(self, source, target):
        """Find all directed paths from source to target (BFS)."""
        if source == target:
            return [[source]]
        paths = []
        queue = [[source]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for child in self.children[node]:
                new_path = path + [child]
                if child == target:
                    paths.append(new_path)
                else:
                    queue.append(new_path)
        return paths

    def causal_effect_vars(self, target="y"):
        """
        Return variables that have a directed causal path to target.
        These are the variables whose interventions can change the target.
        """
        return self.ancestors(target)

    def d_separated(self, x, y, conditioning_set):
        """
        Simplified d-separation check.
        For our pre-specified DAG, we mainly use this to determine
        which variable pairs are conditionally independent given mediators.
        """
        # For the CBO kernel, the practical question is:
        # given the causal structure, which input variables have
        # independent effects on y?
        x_paths = self.get_causal_path(x, "y")
        y_paths = self.get_causal_path(y, "y")
        # If they share no mediator, they are independent
        x_mediators = set()
        for path in x_paths:
            x_mediators.update(path[1:-1])
        y_mediators = set()
        for path in y_paths:
            y_mediators.update(path[1:-1])
        shared = x_mediators & y_mediators
        return len(shared) == 0

    def get_independent_groups(self, variables, target="y"):
        """
        Partition input variables into groups that affect y through
        independent causal pathways. Variables in the same group
        share mediators; variables in different groups do not.
        Used for kernel decomposition: K = K_group1 + K_group2 + ...
        """
        # Build mediator sets for each variable
        mediator_sets = {}
        for var in variables:
            paths = self.get_causal_path(var, target)
            mediators = set()
            for path in paths:
                mediators.update(path[1:-1])
            mediator_sets[var] = mediators

        # Union-find grouping
        groups = {var: var for var in variables}

        def find(x):
            while groups[x] != x:
                groups[x] = groups[groups[x]]
                x = groups[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                groups[ra] = rb

        for v1 in variables:
            for v2 in variables:
                if v1 < v2 and mediator_sets[v1] & mediator_sets[v2]:
                    union(v1, v2)

        # Collect groups
        result = defaultdict(list)
        for var in variables:
            result[find(var)].append(var)
        return list(result.values())

    def causal_strength_prior(self):
        """
        Return prior beliefs about relative causal effect strength,
        based on code analysis. Used to initialize GP lengthscales.
        Larger value = stronger expected effect on y.
        """
        return {
            # Strong effects (from Sobol analysis + code logic)
            "n_red":    0.9,
            "n_blue":   0.8,
            "red_ai":   0.85,
            "max_phases": 0.6,
            # Moderate effects
            "p_urban":  0.4,
            "blue_side": 0.3,
            # Weak effects
            "p_rough":  0.2,
            "p_marsh":  0.2,
            "scenario_seed": 0.1,
        }

    def summary(self):
        """Print DAG summary."""
        print(f"Causal DAG: {len(self.nodes)} nodes, "
              f"{sum(len(v) for v in self.children.values())} edges")
        causal_vars = self.causal_effect_vars("y")
        print(f"Variables with causal path to y: {causal_vars}")
        all_inputs = list(config.DECISION_VARS.keys()) + \
                     list(config.ADVERSARIAL_VARS.keys())
        groups = self.get_independent_groups(
            [v for v in all_inputs if v in causal_vars])
        print(f"Independent causal groups: {groups}")


if __name__ == "__main__":
    dag = CausalDAG()
    dag.summary()
    print("\nCausal paths from n_red to y:")
    for path in dag.get_causal_path("n_red", "y"):
        print("  -> ".join(path))
    print("\nCausal paths from p_urban to y:")
    for path in dag.get_causal_path("p_urban", "y"):
        print("  -> ".join(path))
