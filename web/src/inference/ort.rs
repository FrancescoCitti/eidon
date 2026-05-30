use wasm_bindgen::prelude::*;

// Inline JS bridge to onnxruntime-web (globalThis.ort loaded via <script> in index.html).
// inline_js embeds this directly in the wasm-bindgen glue — no separate file needed.
#[wasm_bindgen(inline_js = r#"
if (typeof globalThis.ort !== 'undefined') {
    globalThis.ort.env.wasm.wasmPaths =
        'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    // Disable multi-threading — requires SharedArrayBuffer (COOP/COEP headers)
    // which trunk's dev server does not set.
    globalThis.ort.env.wasm.numThreads = 1;
}

export async function ort_create_session(model_url) {
    console.log('[ort] fetching', model_url);
    const resp = await fetch(model_url);
    if (!resp.ok) throw new Error('fetch ' + resp.status + ' ' + model_url);
    const buffer = await resp.arrayBuffer();
    const b = new Uint8Array(buffer);
    console.log('[ort] model bytes:', b.length, 'first8:', b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
    return await globalThis.ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
    });
}

export function ort_input_name(session) {
    return session.inputNames[0];
}

export async function ort_run(session, input_name, data, dims) {
    const tensor = new globalThis.ort.Tensor('float32', data, dims);
    const result = await session.run({ [input_name]: tensor });
    return session.outputNames.map(n => result[n].data);
}
"#)]
extern "C" {
    /// Create an ONNX InferenceSession by URL — ort-web fetches the model itself.
    #[wasm_bindgen(catch)]
    pub async fn ort_create_session(model_url: &str) -> Result<JsValue, JsValue>;

    /// Return the first input node name for the session.
    pub fn ort_input_name(session: &JsValue) -> String;

    /// Run inference; returns a JS Array of Float32Arrays (one per output, ordered).
    #[wasm_bindgen(catch)]
    pub async fn ort_run(
        session: &JsValue,
        input_name: &str,
        data: &js_sys::Float32Array,
        dims: &js_sys::Array,
    ) -> Result<JsValue, JsValue>;
}

/// Extract the `idx`-th output tensor as a flat `Vec<f32>`.
pub fn get_output(outputs: &JsValue, idx: u32) -> Vec<f32> {
    use wasm_bindgen::JsCast;
    let arr = js_sys::Array::from(outputs);
    arr.get(idx)
        .unchecked_into::<js_sys::Float32Array>()
        .to_vec()
}

/// Build a JS `Array` of numbers from a Rust slice — used for tensor shape dims.
pub fn dims_array(shape: &[u32]) -> js_sys::Array {
    shape.iter().map(|&d| JsValue::from(d)).collect()
}
