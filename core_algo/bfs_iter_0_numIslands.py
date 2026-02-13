from collections import deque

def numIslands(grid):
    rows, cols = len(grid), len(grid[0])
    seen = set()
    res = 0

    def bfs(r,c):
        q = deque()
        seen.add((r,c))
        q.append((r,c))

        while q:
            row,col = q.popleft()
            directions = [(-1,0),(1,0),(0,-1),(0,1)]
            for d in directions:
                r_new = row+d[0]
                c_new = col+d[1]
                if r_new in range(rows) and c_new in range(cols) and grid[r_new][c_new]=="1" and (r_new,c_new) not in seen:
                    q.append((r_new,c_new))
                    seen.add((r_new,c_new))

    for i in range(rows):
        for j in range(cols):
            if grid[i][j]=="1" and (i,j) not in seen:
                bfs(i,j)
                res +=1

    return res 

if __name__ == "__main__":
    grid = [
    ["1","1","0","0","1"],
    ["1","1","0","0","1"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
    ]
    numIslands(grid)