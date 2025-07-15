from src.utils import compute_clip_embedding, cosine_similarity

class PlayerTracker:
    def __init__(self):
        self.next_id = 0
        self.active_players = {}  # track_id -> (bbox, clip_feature)
        self.history = {}         # track_id -> (bbox, clip_feature)

    def update(self, detections):
        matched = {}
        new_active = {}
        used_ids = set()

        for bbox, crop in detections:
            feat = compute_clip_embedding(crop)
            best_id = None
            best_sim = 0

            # Match against current active players
            for track_id, (_, prev_feat) in self.active_players.items():
                sim = cosine_similarity(feat, prev_feat)
                print(f"[ACTIVE] Trying to match: Sim={sim:.2f} with ID {track_id}")
                if sim > best_sim and sim > 0.75:  # lowered threshold
                    best_id = track_id
                    best_sim = sim

            # Match against history
            if best_id is None:
                for track_id, (_, hist_feat) in self.history.items():
                    if track_id in used_ids:
                        continue
                    sim = cosine_similarity(feat, hist_feat)
                    print(f"[HISTORY] Trying to match: Sim={sim:.2f} with ID {track_id}")
                    if sim > best_sim and sim > 0.75:
                        best_id = track_id
                        best_sim = sim

            # Assign new or existing ID
            if best_id is not None:
                matched[best_id] = bbox
                new_active[best_id] = (bbox, feat)
                used_ids.add(best_id)
            else:
                matched[self.next_id] = bbox
                new_active[self.next_id] = (bbox, feat)
                print(f"[NEW] Assigning new ID: {self.next_id}")
                used_ids.add(self.next_id)
                self.next_id += 1

        # Update long-term memory
        for track_id, (bbox, feat) in new_active.items():
            self.history[track_id] = (bbox, feat)

        self.active_players = new_active
        return matched
