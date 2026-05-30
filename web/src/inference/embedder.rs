/// ArcFace MobileNet embedder (w600k_mbf.onnx from InsightFace buffalo_s).
///
/// Preprocessing: 112×112 aligned RGB float32 → (pixel − 127.5) / 128.0 → CHW → NCHW.

use wasm_bindgen::JsValue;
use wasm_bindgen::JsCast;

use super::ort::{ort_create_session, ort_input_name, ort_run, get_output, dims_array};

const EMBED_SIZE: usize = 112;
pub const EMBED_DIM: usize = 512;

pub fn l2_normalize(data: &mut Vec<f32>, n: usize, d: usize) {
    for i in 0..n {
        let sq: f32 = data[i * d..(i + 1) * d].iter().map(|&x| x * x).sum();
        let norm = sq.sqrt().max(1e-10);
        for j in 0..d {
            data[i * d + j] /= norm;
        }
    }
}

fn preprocess_crop(crop: &web_sys::HtmlCanvasElement) -> Result<Vec<f32>, JsValue> {
    let ctx = crop
        .get_context("2d")?
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;
    let img = ctx.get_image_data(0.0, 0.0, EMBED_SIZE as f64, EMBED_SIZE as f64)?;
    let rgba = img.data();

    let hw = EMBED_SIZE * EMBED_SIZE;
    let mut nchw = vec![0.0f32; 3 * hw];
    for y in 0..EMBED_SIZE {
        for x in 0..EMBED_SIZE {
            let src_idx = (y * EMBED_SIZE + x) * 4;
            let px = y * EMBED_SIZE + x;
            nchw[px]          = (rgba[src_idx + 0] as f32 - 127.5) / 128.0;
            nchw[hw + px]     = (rgba[src_idx + 1] as f32 - 127.5) / 128.0;
            nchw[2 * hw + px] = (rgba[src_idx + 2] as f32 - 127.5) / 128.0;
        }
    }
    Ok(nchw)
}

/// ArcFace MobileNet embedder backed by onnxruntime-web.
pub struct Embedder {
    session: JsValue,
    input_name: String,
}

impl Embedder {
    pub async fn from_url(url: &str) -> Result<Self, String> {
        let session = ort_create_session(url)
            .await
            .map_err(|e| format!("ort session: {:?}", e))?;
        let input_name = ort_input_name(&session);
        Ok(Self { session, input_name })
    }

    /// Embed one aligned 112×112 canvas crop.
    /// Returns an `EMBED_DIM`-element L2-normalised vector.
    pub async fn embed(
        &self,
        crop: &web_sys::HtmlCanvasElement,
    ) -> Result<Vec<f32>, String> {
        let nchw = preprocess_crop(crop).map_err(|e| format!("{:?}", e))?;

        let data = js_sys::Float32Array::from(nchw.as_slice());
        let dims = dims_array(&[1, 3, EMBED_SIZE as u32, EMBED_SIZE as u32]);

        let outputs = ort_run(&self.session, &self.input_name, &data, &dims)
            .await
            .map_err(|e| format!("ort run: {:?}", e))?;

        let mut raw = get_output(&outputs, 0);
        l2_normalize(&mut raw, 1, EMBED_DIM);
        Ok(raw)
    }
}
