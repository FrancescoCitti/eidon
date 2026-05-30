/// Intersection-over-union of two \[x1,y1,x2,y2\] boxes stored in flat slices.
fn iou(boxes: &[[f32; 4]], i: usize, j: usize) -> f32 {
    let [ax1, ay1, ax2, ay2] = boxes[i];
    let [bx1, by1, bx2, by2] = boxes[j];

    let inter_w = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let inter_h = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = inter_w * inter_h;
    if inter == 0.0 {
        return 0.0;
    }
    let a_area = (ax2 - ax1) * (ay2 - ay1);
    let b_area = (bx2 - bx1) * (by2 - by1);
    inter / (a_area + b_area - inter)
}

/// Non-maximum suppression.
///
/// `boxes` — slice of \[x1,y1,x2,y2\] boxes.
/// `scores` — per-box confidence scores (same length as `boxes`).
/// Returns indices of kept boxes, highest-score first.
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut suppressed = vec![false; scores.len()];
    let mut kept = Vec::new();

    for &i in &order {
        if suppressed[i] { continue; }
        kept.push(i);
        for &j in &order {
            if suppressed[j] || j == i { continue; }
            if iou(boxes, i, j) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    kept
}
