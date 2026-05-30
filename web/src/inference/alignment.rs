/// ArcFace 5-landmark destination template for a 112×112 output.
/// Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Compute the 2×3 similarity-transform matrix that maps `src_pts` onto `dst_pts`
/// in the least-squares sense (closed-form, equivalent to Umeyama for this layout).
///
/// Returns `[m00, m01, m02, m10, m11, m12]` row-major:
///   x' = m00·x + m01·y + m02
///   y' = m10·x + m11·y + m12
pub fn similarity_transform(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [f64; 6] {
    let n = 5.0f64;

    let mut xs_mean = 0.0f64;
    let mut ys_mean = 0.0f64;
    let mut xd_mean = 0.0f64;
    let mut yd_mean = 0.0f64;
    for i in 0..5 {
        xs_mean += src[i][0] as f64;
        ys_mean += src[i][1] as f64;
        xd_mean += dst[i][0] as f64;
        yd_mean += dst[i][1] as f64;
    }
    xs_mean /= n; ys_mean /= n; xd_mean /= n; yd_mean /= n;

    let mut src_var = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..5 {
        let xs0 = src[i][0] as f64 - xs_mean;
        let ys0 = src[i][1] as f64 - ys_mean;
        let xd0 = dst[i][0] as f64 - xd_mean;
        let yd0 = dst[i][1] as f64 - yd_mean;
        src_var += xs0 * xs0 + ys0 * ys0;
        sum_a += xs0 * xd0 + ys0 * yd0;
        sum_b += xs0 * yd0 - ys0 * xd0;
    }
    src_var /= n;

    let d = n * src_var;
    let a = sum_a / d;  // scale · cos θ
    let b = sum_b / d;  // scale · sin θ

    let tx = xd_mean - a * xs_mean + b * ys_mean;
    let ty = yd_mean - b * xs_mean - a * ys_mean;

    // Row 0: [a, -b, tx],  Row 1: [b, a, ty]
    [a, -b, tx, b, a, ty]
}

/// Affine-warp a video frame to the 112×112 ArcFace template using Canvas 2D.
///
/// Applies the forward similarity transform with `ctx.setTransform`, so the
/// facial keypoints in the source frame land exactly at the ArcFace template
/// positions in the output 112×112 canvas.
pub fn align_face_image(
    source: &web_sys::HtmlImageElement,
    landmarks: &[f32; 10],
    size: u32,
) -> Result<web_sys::HtmlCanvasElement, wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;
    let scale = size as f64 / 112.0;
    let src_pts = [
        [landmarks[0], landmarks[1]], [landmarks[2], landmarks[3]],
        [landmarks[4], landmarks[5]], [landmarks[6], landmarks[7]],
        [landmarks[8], landmarks[9]],
    ];
    let dst_pts: [[f32; 2]; 5] = std::array::from_fn(|i| {
        [ARCFACE_DST[i][0] * scale as f32, ARCFACE_DST[i][1] * scale as f32]
    });
    let m = similarity_transform(&src_pts, &dst_pts);
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.create_element("canvas")?.dyn_into::<web_sys::HtmlCanvasElement>()?;
    canvas.set_width(size);
    canvas.set_height(size);
    let ctx = canvas.get_context("2d")?.unwrap().dyn_into::<web_sys::CanvasRenderingContext2d>()?;
    ctx.set_transform(m[0], m[3], m[1], m[4], m[2], m[5])?;
    ctx.draw_image_with_html_image_element(source, 0.0, 0.0)?;
    ctx.reset_transform()?;
    Ok(canvas)
}

pub fn align_face(
    source: &web_sys::HtmlVideoElement,
    landmarks: &[f32; 10],
    size: u32,
) -> Result<web_sys::HtmlCanvasElement, wasm_bindgen::JsValue> {
    use wasm_bindgen::JsCast;

    let scale = size as f64 / 112.0;
    let src_pts = [
        [landmarks[0], landmarks[1]],
        [landmarks[2], landmarks[3]],
        [landmarks[4], landmarks[5]],
        [landmarks[6], landmarks[7]],
        [landmarks[8], landmarks[9]],
    ];
    let dst_pts: [[f32; 2]; 5] = std::array::from_fn(|i| {
        [ARCFACE_DST[i][0] * scale as f32, ARCFACE_DST[i][1] * scale as f32]
    });

    let m = similarity_transform(&src_pts, &dst_pts);
    // M = [m00, m01, m02, m10, m11, m12]
    // Canvas setTransform(a, b, c, d, e, f):
    //   x' = a·x + c·y + e,  y' = b·x + d·y + f
    // → a=m[0], b=m[3], c=m[1], d=m[4], e=m[2], f=m[5]

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    canvas.set_width(size);
    canvas.set_height(size);

    let ctx = canvas
        .get_context("2d")?
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

    ctx.set_transform(m[0], m[3], m[1], m[4], m[2], m[5])?;
    ctx.draw_image_with_html_video_element(source, 0.0, 0.0)?;
    ctx.reset_transform()?;

    Ok(canvas)
}
