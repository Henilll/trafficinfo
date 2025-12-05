import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# -----------------------
# SAFE IoU
# -----------------------
def iou(bb_test, bb_gt):
    """Computes IoU between two boxes"""
    if bb_test[2] <= bb_test[0] or bb_test[3] <= bb_test[1]:
        return 0.0
    if bb_gt[2] <= bb_gt[0] or bb_gt[3] <= bb_gt[1]:
        return 0.0

    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h

    area1 = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1])

    return inter / (area1 + area2 - inter + 1e-6)

# -----------------------
# Conversion
# -----------------------
def convert_bbox_to_z(bbox):
    w = max(1., bbox[2] - bbox[0])
    h = max(1., bbox[3] - bbox[1])
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    eps = 1e-6
    s = max(eps, x[2])
    r = max(eps, x[3])
    w = np.sqrt(s * r)
    h = s / max(w, eps)
    if np.isnan(w) or np.isnan(h):
        return np.array([0,0,0,0]).reshape((1,4))
    x1 = x[0] - w/2.
    y1 = x[1] - h/2.
    x2 = x[0] + w/2.
    y2 = x[1] + h/2.
    return np.array([x1, y1, x2, y2]).reshape((1,4))

# -----------------------
# Tracker Class
# -----------------------
class KalmanBoxTracker:
    _count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.R *= 1.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.hits = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

    def update(self, bbox):
        self.kf.update(convert_bbox_to_z(bbox))
        self.time_since_update = 0
        self.hits += 1

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        bbox = convert_x_to_bbox(self.kf.x)
        if np.any(np.isnan(bbox)):
            return np.array([0,0,0,0])
        return bbox[0]

# -----------------------
# SORT Main
# -----------------------
class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        outputs = []
        dets = np.array(dets)
        if len(dets) == 0:
            dets = np.empty((0,5))

        # Predict
        predicted_boxes = np.array([trk.predict() for trk in self.trackers])
        if len(predicted_boxes) == 0:
            predicted_boxes = np.empty((0,4))

        # Match using Hungarian algorithm
        matched, unmatched_dets, unmatched_trks = associate(predicted_boxes, dets, self.iou_threshold)

        # Update matched trackers
        for t, d in matched:
            self.trackers[t].update(dets[d])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i]))

        # Remove dead trackers & prepare output
        alive = []
        for trk in self.trackers:
            if trk.time_since_update < self.max_age:
                bbox = convert_x_to_bbox(trk.kf.x)[0]
                if (bbox[2]-bbox[0] > 1) and (bbox[3]-bbox[1] > 1):
                    outputs.append(np.append(bbox, trk.id))
                    alive.append(trk)

        self.trackers = alive
        return np.array(outputs)

# -----------------------
# Hungarian IoU Matching
# -----------------------
def associate(preds, dets, thresh):
    if len(preds) == 0:
        return [], list(range(len(dets))), []

    iou_matrix = np.zeros((len(preds), len(dets)))
    for t, trk in enumerate(preds):
        for d, det in enumerate(dets):
            iou_matrix[t,d] = iou(trk, det)

    matched_idx = linear_sum_assignment(-iou_matrix)  # maximize IoU
    matched = []
    for t, d in zip(*matched_idx):
        if iou_matrix[t,d] >= thresh:
            matched.append((t,d))

    unmatched_t = [t for t in range(len(preds)) if t not in [m[0] for m in matched]]
    unmatched_d = [d for d in range(len(dets)) if d not in [m[1] for m in matched]]

    return matched, unmatched_d, unmatched_t
