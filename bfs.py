from collections import deque


def find_path_to_target_adjacent_bidirectional(n, m, obstacles, start, target):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    obstacle_set = set(obstacles)

    def get_neighbors(x, y):
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in obstacle_set:
                yield (nx, ny)

    # Set of valid adjacent cells to target
    target_adjacents = set(get_neighbors(*target))
    if not target_adjacents:
        return None  # No adjacent cells to target are accessible

    if start in target_adjacents:
        return [start]  # Already adjacent

    # BFS from start
    queue_start = deque([start])
    parents_start = {start: None}

    # BFS from each target-adjacent cell
    queue_goal = deque(target_adjacents)
    parents_goal = {pos: None for pos in target_adjacents}

    visited_start = set([start])
    visited_goal = set(target_adjacents)

    while queue_start and queue_goal:
        # Expand from start side
        for _ in range(len(queue_start)):
            current = queue_start.popleft()
            for neighbor in get_neighbors(*current):
                if neighbor in visited_start:
                    continue
                parents_start[neighbor] = current
                visited_start.add(neighbor)
                queue_start.append(neighbor)

                if neighbor in visited_goal:
                    # Meeting point found
                    return reconstruct_path(parents_start, parents_goal, neighbor)

        # Expand from goal side
        for _ in range(len(queue_goal)):
            current = queue_goal.popleft()
            for neighbor in get_neighbors(*current):
                if neighbor in visited_goal:
                    continue
                parents_goal[neighbor] = current
                visited_goal.add(neighbor)
                queue_goal.append(neighbor)

                if neighbor in visited_start:
                    # Meeting point found
                    return reconstruct_path(parents_start, parents_goal, neighbor)

    return None  # No path found

def reconstruct_path(parents_start, parents_goal, meeting_point):
    # Build path from start to meeting point
    path_start = []
    node = meeting_point
    while node:
        path_start.append(node)
        node = parents_start[node]
    path_start.reverse()

    # Build path from meeting point to goal (adjacent)
    path_goal = []
    node = parents_goal[meeting_point]
    while node:
        path_goal.append(node)
        node = parents_goal[node]

    return path_start + path_goal
