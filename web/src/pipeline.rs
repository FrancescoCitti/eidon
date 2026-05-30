/// Orchestrates detect → align → embed → match, mirroring the Python pipeline.

use std::collections::HashMap;
use crate::inference::{alignment::{align_face, align_face_image}, detector::Detector, embedder::Embedder};
use crate::gallery::gallery_get_all;
use crate::types::{Detection, RecognitionResult};

const UNKNOWN: &str = "unknown";
pub use crate::inference::embedder::EMBED_DIM;

/// Cosine similarity of two L2-normalised `EMBED_DIM`-vectors = plain dot product.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Detection with the largest bounding-box area.
fn largest(detections: &[Detection]) -> &Detection {
    detections
        .iter()
        .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap())
        .unwrap()
}

/// Loaded, ready-to-use recognition pipeline.
pub struct Pipeline {
    pub detector: Detector,
    pub embedder: Embedder,
}

impl Pipeline {
    /// Run the full pipeline: detect → align → embed → match gallery.
    ///
    /// `threshold` — minimum cosine similarity to accept a match (default 0.40).
    pub async fn recognize(
        &self,
        source: &web_sys::HtmlVideoElement,
        threshold: f32,
    ) -> Result<Vec<RecognitionResult>, String> {
        let detections = self
            .detector
            .detect(source, 0.5, 0.4)
            .await?;

        if detections.is_empty() {
            return Ok(Vec::new());
        }

        let gallery: HashMap<String, Vec<Vec<f32>>> = gallery_get_all().await?;
        let mut results = Vec::with_capacity(detections.len());

        for det in &detections {
            let crop = align_face(source, &det.landmarks, 112)
                .map_err(|e| format!("{:?}", e))?;
            let embedding = self.embedder.embed(&crop).await?;

            let mut best_identity = UNKNOWN.to_owned();
            let mut best_sim = -1.0f32;

            for (identity, embeddings) in &gallery {
                for stored in embeddings {
                    let sim = cosine_sim(&embedding, stored);
                    if sim > best_sim {
                        best_sim = sim;
                        best_identity = identity.clone();
                    }
                }
            }

            let matched = best_sim >= threshold;
            results.push(RecognitionResult {
                detection: det.clone(),
                identity: if matched { best_identity } else { UNKNOWN.to_owned() },
                similarity: best_sim,
                matched,
            });
        }

        Ok(results)
    }

    /// Detect the largest face and enroll it under `identity`.
    /// Returns `true` if a face was found.
    /// Detect the largest face in a still image and enroll it.
    pub async fn enroll_image(
        &self,
        source: &web_sys::HtmlImageElement,
        identity: &str,
    ) -> Result<bool, String> {
        let detections = self.detector.detect_image(source, 0.6, 0.4).await?;
        if detections.is_empty() { return Ok(false); }
        let det = largest(&detections);
        let crop = align_face_image(source, &det.landmarks, 112)
            .map_err(|e| format!("{:?}", e))?;
        let embedding = self.embedder.embed(&crop).await?;
        crate::gallery::gallery_add(identity, embedding).await?;
        Ok(true)
    }

    pub async fn enroll(
        &self,
        source: &web_sys::HtmlVideoElement,
        identity: &str,
    ) -> Result<bool, String> {
        let detections = self
            .detector
            .detect(source, 0.6, 0.4)
            .await?;

        if detections.is_empty() {
            return Ok(false);
        }

        let det = largest(&detections);
        let crop = align_face(source, &det.landmarks, 112)
            .map_err(|e| format!("{:?}", e))?;
        let embedding = self.embedder.embed(&crop).await?;
        crate::gallery::gallery_add(identity, embedding).await?;
        Ok(true)
    }
}
