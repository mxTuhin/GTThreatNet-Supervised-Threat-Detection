import math
from collections import defaultdict, deque

class TrajectoryStore:
    def __init__(self, maxlen=60):
        self.maxlen = maxlen
        self.tracks = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id, frame_idx, cx, cy):
        self.tracks[track_id].append((frame_idx, cx, cy))

    def get_points(self, track_id):
        return list(self.tracks.get(track_id, []))

    def get_all_ids(self):
        return list(self.tracks.keys())

def vector_from_points(points):
    """
    Returns approximate movement vector from first to last point.
    """
    if len(points) < 2:
        return None

    _, x1, y1 = points[0]
    _, x2, y2 = points[-1]
    return (x2 - x1, y2 - y1)

def norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)

def cosine_similarity(v1, v2):
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)

def average_distance(points_a, points_b):
    """
    Compute average distance on overlapping suffix length.
    """
    n = min(len(points_a), len(points_b))
    if n == 0:
        return None

    pa = points_a[-n:]
    pb = points_b[-n:]

    distances = []
    for (_, ax, ay), (_, bx, by) in zip(pa, pb):
        d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        distances.append(d)

    return sum(distances) / len(distances)

def behind_ratio(points_leader, points_follower):
    """
    Naive approximation:
    If movement is mostly downward/upward/left/right,
    check whether follower remains behind leader along that axis.
    """
    n = min(len(points_leader), len(points_follower))
    if n < 5:
        return 0.0

    leader = points_leader[-n:]
    follower = points_follower[-n:]

    v = vector_from_points(leader)
    if v is None:
        return 0.0

    vx, vy = v
    count_behind = 0

    for (_, lx, ly), (_, fx, fy) in zip(leader, follower):
        # Determine dominant direction
        if abs(vx) >= abs(vy):
            # Horizontal motion
            if vx > 0 and fx < lx:
                count_behind += 1
            elif vx < 0 and fx > lx:
                count_behind += 1
        else:
            # Vertical motion
            if vy > 0 and fy < ly:
                count_behind += 1
            elif vy < 0 and fy > ly:
                count_behind += 1

    return count_behind / n