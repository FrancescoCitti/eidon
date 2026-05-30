/// A face detected in one frame.
#[derive(Debug, Clone)]
pub struct Detection {
    /// \[x1, y1, x2, y2\] in source-image pixels.
    pub bbox: [f32; 4],
    /// Flat \[x0,y0, x1,y1, x2,y2, x3,y3, x4,y4\] — 5 facial keypoints.
    /// Order: left\_eye, right\_eye, nose\_tip, left\_mouth, right\_mouth.
    pub landmarks: [f32; 10],
    /// RetinaFace detection confidence ∈ \[0, 1\].
    pub score: f32,
}

impl Detection {
    pub fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    }
}

/// Identity matching result for one embedding query.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Best-match identity name, or `"unknown"` when below threshold.
    pub identity: String,
    /// Cosine similarity to the nearest gallery embedding.
    pub similarity: f32,
    /// `true` when `similarity >= configured threshold`.
    pub matched: bool,
}

/// Full pipeline result for one detected face.
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    pub detection: Detection,
    pub identity: String,
    pub similarity: f32,
    pub matched: bool,
}
