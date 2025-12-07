# pose_geometry.py

import numpy as np
import math
from collections import defaultdict


class PoseGeometry:
    """
    Handles:
      - angle computation
      - coordinate normalization (root+scale)
      - building angle triplets from pose connections
    """

    def __init__(self, mp_pose, exclude_angle_centers=None, eps=1e-6):
        self.mp_pose = mp_pose
        self.pose_connections = mp_pose.POSE_CONNECTIONS
        self.eps = eps

        if exclude_angle_centers is None:
            # your original exclusion: face (0-10), wrists/hands (15-22)
            exclude_angle_centers = list(range(0, 11)) + list(range(15, 23))
        self.exclude_angle_centers = exclude_angle_centers

        self.angle_triplets = self._build_angle_triplets()
        self.num_angle_features = len(self.angle_triplets)

    # ---------- core math ----------

    @staticmethod
    def compute_angle(a, b, c):
        """
        Angle at point b formed by vectors ba and bc, returns radians.
        a, b, c are np.array([x,y]).
        """
        ba = a - b
        bc = c - b
        na = np.linalg.norm(ba) + 1e-8
        nb = np.linalg.norm(bc) + 1e-8
        ba /= na
        bc /= nb
        cosang = np.clip(np.dot(ba, bc), -1.0, 1.0)
        return math.acos(cosang)

    def normalize_coords(self, coords):
        """
        coords: (V,2) array of MediaPipe normalized coords (0..1, 0..1)
        Returns root-relative, scale-normalized coords (V,2).

        Steps:
          1. Root joint: mid-hip if possible, else mean of all joints
          2. Compute coords relative to root
          3. Compute scale: average bone length of skeleton edges
          4. Divide coords by scale
        """
        V = coords.shape[0]

        # 1) Get root joint: mid-hip if exists
        L_HIP = 23
        R_HIP = 24

        if L_HIP < V and R_HIP < V:
            root = (coords[L_HIP] + coords[R_HIP]) / 2.0
        else:
            # fallback: center of mass
            root = coords.mean(axis=0)

        coords_rel = coords - root

        # 2) compute scale factor based on bone lengths
        lengths = []
        for (a, b) in self.pose_connections:
            if a < V and b < V:
                d = np.linalg.norm(coords[a] - coords[b])
                if d > self.eps:
                    lengths.append(d)

        if len(lengths) == 0:
            scale = 1.0
        else:
            scale = float(np.mean(lengths))

        if scale < self.eps:
            scale = 1.0

        coords_norm = coords_rel / scale
        return coords_norm

    # ---------- angle triplets ----------

    def _build_angle_triplets(self):
        """
        Build angle triplets (a,b,c) from POSE_CONNECTIONS.
        b is the 'center' joint; exclude_angle_centers are skipped.
        """
        adj = defaultdict(set)
        for a, c in self.pose_connections:
            adj[a].add(c)
            adj[c].add(a)

        triplets = []
        for b, neighbors in adj.items():
            if b in self.exclude_angle_centers:
                continue
            neighbors = list(neighbors)
            if len(neighbors) < 2:
                continue
            n = len(neighbors)
            for i in range(n):
                for j in range(i + 1, n):
                    a = neighbors[i]
                    c = neighbors[j]
                    triplets.append((a, b, c))
        return triplets
