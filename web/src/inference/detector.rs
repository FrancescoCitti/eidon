/// SCRFD RetinaFace detector (det_500m.onnx from InsightFace buffalo_s).
///
/// Model output layout (9 tensors for 3 strides × 3 heads):
///   outputs[0..2]  — confidence scores  per stride (8, 16, 32)
///   outputs[3..5]  — bbox distance preds per stride (× stride = pixel units)
///   outputs[6..8]  — keypoint offsets   per stride (× stride = pixel units)
///
/// Preprocessing: (pixel − 127.5) / 128.0, RGB channel order, NCHW float32.

use wasm_bindgen::JsValue;
use wasm_bindgen::JsCast;

use crate::types::Detection;
use super::nms::nms;
use super::ort::{ort_create_session, ort_input_name, ort_run, get_output, dims_array};

const INPUT_SIZE: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
const ANCHORS_PER_CELL: usize = 2;

fn generate_anchors(h: usize, w: usize, stride: usize) -> Vec<[f32; 2]> {
    let n = h * w * ANCHORS_PER_CELL;
    let mut centers = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            for _ in 0..ANCHORS_PER_CELL {
                centers.push([(x * stride) as f32, (y * stride) as f32]);
            }
        }
    }
    centers
}

fn decode_bbox(anchors: &[[f32; 2]], dist: &[f32], stride: usize) -> Vec<[f32; 4]> {
    anchors
        .iter()
        .enumerate()
        .map(|(i, &[cx, cy])| {
            let s = stride as f32;
            [
                cx - dist[i * 4 + 0] * s,
                cy - dist[i * 4 + 1] * s,
                cx + dist[i * 4 + 2] * s,
                cy + dist[i * 4 + 3] * s,
            ]
        })
        .collect()
}

fn decode_kps(anchors: &[[f32; 2]], dist: &[f32], stride: usize) -> Vec<[f32; 10]> {
    anchors
        .iter()
        .enumerate()
        .map(|(i, &[cx, cy])| {
            let s = stride as f32;
            let mut kps = [0.0f32; 10];
            for k in 0..5 {
                kps[k * 2 + 0] = cx + dist[i * 10 + k * 2 + 0] * s;
                kps[k * 2 + 1] = cy + dist[i * 10 + k * 2 + 1] * s;
            }
            kps
        })
        .collect()
}

fn make_preprocess_canvas() -> Result<(web_sys::HtmlCanvasElement, web_sys::CanvasRenderingContext2d), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let tmp = document
        .create_element("canvas")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    tmp.set_width(INPUT_SIZE as u32);
    tmp.set_height(INPUT_SIZE as u32);
    let ctx = tmp
        .get_context("2d")?
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;
    Ok((tmp, ctx))
}

fn read_nchw(ctx: &web_sys::CanvasRenderingContext2d, src_w: f32, src_h: f32) -> Result<(Vec<f32>, f32, f32), JsValue> {
    let img_data = ctx.get_image_data(0.0, 0.0, INPUT_SIZE as f64, INPUT_SIZE as f64)?;
    let rgba = img_data.data();
    let hw = INPUT_SIZE * INPUT_SIZE;
    let mut nchw = vec![0.0f32; 3 * hw];
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let si = (y * INPUT_SIZE + x) * 4;
            let px = y * INPUT_SIZE + x;
            nchw[px]          = (rgba[si + 0] as f32 - 127.5) / 128.0;
            nchw[hw + px]     = (rgba[si + 1] as f32 - 127.5) / 128.0;
            nchw[2 * hw + px] = (rgba[si + 2] as f32 - 127.5) / 128.0;
        }
    }
    Ok((nchw, src_w / INPUT_SIZE as f32, src_h / INPUT_SIZE as f32))
}

fn preprocess(source: &web_sys::HtmlVideoElement) -> Result<(Vec<f32>, f32, f32), JsValue> {
    let src_w = source.video_width() as f32;
    let src_h = source.video_height() as f32;
    let (_, ctx) = make_preprocess_canvas()?;
    ctx.draw_image_with_html_video_element_and_dw_and_dh(
        source, 0.0, 0.0, INPUT_SIZE as f64, INPUT_SIZE as f64,
    )?;
    read_nchw(&ctx, src_w, src_h)
}

fn preprocess_image(source: &web_sys::HtmlImageElement) -> Result<(Vec<f32>, f32, f32), JsValue> {
    let src_w = source.natural_width() as f32;
    let src_h = source.natural_height() as f32;
    let (_, ctx) = make_preprocess_canvas()?;
    ctx.draw_image_with_html_image_element_and_dw_and_dh(
        source, 0.0, 0.0, INPUT_SIZE as f64, INPUT_SIZE as f64,
    )?;
    read_nchw(&ctx, src_w, src_h)
}

/// SCRFD face detector backed by onnxruntime-web.
pub struct Detector {
    session: JsValue,
    input_name: String,
}

impl Detector {
    pub async fn from_url(url: &str) -> Result<Self, String> {
        let session = ort_create_session(url)
            .await
            .map_err(|e| format!("ort session: {:?}", e))?;
        let input_name = ort_input_name(&session);
        Ok(Self { session, input_name })
    }

    pub async fn detect(
        &self,
        source: &web_sys::HtmlVideoElement,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>, String> {
        let prep = preprocess(source).map_err(|e| format!("preprocess: {:?}", e))?;
        self.run_detection(prep, score_threshold, nms_threshold).await
    }

    pub async fn detect_image(
        &self,
        source: &web_sys::HtmlImageElement,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>, String> {
        let prep = preprocess_image(source).map_err(|e| format!("preprocess: {:?}", e))?;
        self.run_detection(prep, score_threshold, nms_threshold).await
    }

    async fn run_detection(
        &self,
        (nchw, scale_x, scale_y): (Vec<f32>, f32, f32),
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<Detection>, String> {

        let data = js_sys::Float32Array::from(nchw.as_slice());
        let dims = dims_array(&[1, 3, INPUT_SIZE as u32, INPUT_SIZE as u32]);

        let outputs = ort_run(&self.session, &self.input_name, &data, &dims)
            .await
            .map_err(|e| format!("ort run: {:?}", e))?;

        let fmc = STRIDES.len();
        let mut candidate_boxes: Vec<[f32; 4]> = Vec::new();
        let mut candidate_scores: Vec<f32> = Vec::new();
        let mut candidate_kps: Vec<[f32; 10]> = Vec::new();

        for (si, &stride) in STRIDES.iter().enumerate() {
            let grid_h = INPUT_SIZE / stride;
            let grid_w = INPUT_SIZE / stride;
            let n = grid_h * grid_w * ANCHORS_PER_CELL;

            let scores     = get_output(&outputs, si as u32);
            let bbox_data  = get_output(&outputs, (si + fmc) as u32);
            let kps_data   = get_output(&outputs, (si + fmc * 2) as u32);

            let anchors = generate_anchors(grid_h, grid_w, stride);
            let boxes = decode_bbox(&anchors, &bbox_data, stride);
            let kps_decoded = decode_kps(&anchors, &kps_data, stride);

            for i in 0..n {
                if scores[i] >= score_threshold {
                    candidate_boxes.push(boxes[i]);
                    candidate_scores.push(scores[i]);
                    candidate_kps.push(kps_decoded[i]);
                }
            }
        }

        if candidate_boxes.is_empty() {
            return Ok(Vec::new());
        }

        let kept = nms(&candidate_boxes, &candidate_scores, nms_threshold);

        Ok(kept
            .into_iter()
            .map(|i| {
                let [x1, y1, x2, y2] = candidate_boxes[i];
                let k = candidate_kps[i];
                Detection {
                    bbox: [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                    landmarks: std::array::from_fn(|j| {
                        k[j] * if j % 2 == 0 { scale_x } else { scale_y }
                    }),
                    score: candidate_scores[i],
                }
            })
            .collect())
    }
}
