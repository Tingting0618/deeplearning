from collections import deque

def pacificAtlantic(heights):
    if not heights or not heights[0]:
        return []
    R, C = len(heights), len(heights[0])
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]

    def bfs(starts):
        vis = [[False]*C for _ in range(R)]
        q = deque()

        for r, c in starts:
            if not vis[r][c]:
                vis[r][c] = True
                q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and not vis[nr][nc]:
                    # reverse condition: can move to equal/higher neighbor
                    if heights[nr][nc] >= heights[r][c]:
                        vis[nr][nc] = True
                        q.append((nr, nc))
        return vis

    pac_starts = [(0, c) for c in range(C)] + [(r, 0) for r in range(R)]
    atl_starts = [(R-1, c) for c in range(C)] + [(r, C-1) for r in range(R)]

    pac = bfs(pac_starts)
    atl = bfs(atl_starts)

    ans = []
    for r in range(R):
        for c in range(C):
            if pac[r][c] and atl[r][c]:
                ans.append([r, c])
    return ans

if __name__ =="__main__":
    print(pacificAtlantic([[1,2,6],[3,5,5]]))