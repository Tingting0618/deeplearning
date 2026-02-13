def canFinish(numCourses, prerequisites):
    preMap = {i: [] for i in range(numCourses)}
    # This creates a graph: key = course value = list of prerequisites for that course

    for crs, pre in prerequisites:
        preMap[crs].append(pre)
    # Builds the graph. # Example [0,1] means: to take 0, you must take 1 first.

    visiting = set()
    # This tracks the current DFS path (the recursion stack). # If a course appears here twice, you have a cycle.

    # DFS logic
    def dfs(crs):
        # 1. Cycle check
        if crs in visiting:
            # You are trying to take a course that is already in progress. # That means a loop like A → B → A. # Cycle found ❌
            return False

        # 2. Base case
        if preMap[crs] == []:
            # No prerequisites left. # This course is safe to take ✅
            return True

        # 3. Explore prerequisites
        visiting.add(crs) # Mark this course as currently being explored.

        for pre in preMap[crs]:
            # Try to complete all prerequisites first. # If any prerequisite fails, this course fails too.
            if not dfs(pre):
                return False

        # 4. Cleanup after DFS
        visiting.remove(crs) # Remove from visiting because DFS path is done
        preMap[crs] = []# Set preMap[crs] = [] to memoize the result This means “we already proved this course is safe” # This avoids redoing work.

        return True

    # Final loop
    for c in range(numCourses):# You must check every course, because the graph may be disconnected.
        if not dfs(c):# If any course is part of a cycle → impossible.
            return False
    return True

if __name__ =="__main__":
    print(canFinish(5 ,  [[1, 0],[2, 0],[3, 1], [4, 2]] ) )
