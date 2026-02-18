from typing import Optional

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None

        clones = {}  # original_node -> cloned_node

        def dfs(curr: 'Node') -> 'Node':
            # 1) If already cloned, reuse it (prevents cycles + duplicates)
            if curr in clones:
                return clones[curr]

            # 2) Create clone and store it immediately
            copy = Node(curr.val)
            clones[curr] = copy

            # 3) Clone neighbors and connect them
            for nei in curr.neighbors:
                copy.neighbors.append(dfs(nei))

            return copy

        return dfs(node)
if __name__ =="__main__":
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)

    n1.neighbors = [n2]
    n2.neighbors = [n1, n3]
    n3.neighbors = [n2]

    sol = Solution()
    cloned = sol.cloneGraph(n1)
    print(cloned)