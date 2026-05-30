// Bridge to onnxruntime-web (globalThis.ort loaded via <script> in index.html).
// This module is imported by wasm-bindgen glue; globalThis.ort is always defined
// before the WASM module initialises because the CDN <script> is render-blocking.

// Point ort-web's WASM backend to the same CDN version as the JS bundle so it
// can load its .wasm files regardless of where the page is hosted.
globalThis.ort.env.wasm.wasmPaths =
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

export async function ort_create_session(model_bytes) {
    return await globalThis.ort.InferenceSession.create(model_bytes);
}

export function ort_input_name(session) {
    return session.inputNames[0];
}

// Returns a JS Array of Float32Arrays — one per output, in outputNames order.
export async function ort_run(session, input_name, data, dims) {
    const tensor = new globalThis.ort.Tensor('float32', data, dims);
    const result = await session.run({ [input_name]: tensor });
    return session.outputNames.map(n => result[n].data);
}
