use leptos::*;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

use crate::pipeline::Pipeline;
use crate::types::RecognitionResult;

const GREEN: &str = "#2ea043";
const ORANGE: &str = "#d97706";
const SHADOW: &str = "rgba(0,0,0,0.65)";

// Yields to the browser at the display refresh rate (~60 fps).
#[wasm_bindgen::prelude::wasm_bindgen(inline_js = "
export function raf_promise() {
    return new Promise(r => requestAnimationFrame(() => r(null)));
}
")]
extern "C" {
    async fn raf_promise();
}

fn draw_overlay(
    canvas: &web_sys::HtmlCanvasElement,
    results: &[RecognitionResult],
    fps: f64,
) {
    let Ok(Some(obj)) = canvas.get_context("2d") else { return };
    let Ok(ctx) = obj.dyn_into::<web_sys::CanvasRenderingContext2d>() else { return };

    let w = canvas.width() as f64;
    ctx.clear_rect(0.0, 0.0, w, canvas.height() as f64);

    for r in results {
        let [ox1, y1, ox2, y2] = r.detection.bbox.map(|v| v as f64);
        let x1 = w - ox2;
        let x2 = w - ox1;
        let color = if r.matched { GREEN } else { ORANGE };
        ctx.set_stroke_style(&color.into());
        ctx.set_line_width(2.0);
        ctx.stroke_rect(x1, y1, x2 - x1, y2 - y1);
        let label = if r.matched {
            format!("{}  {:.0}%", r.identity, r.similarity * 100.0)
        } else {
            "Unknown".into()
        };
        draw_label(&ctx, &label, x1, y1, color, true);
    }

    draw_label(&ctx, &format!("{fps:.0} fps"), 8.0, 8.0, "#c0c8d8", false);
}

fn draw_label(
    ctx: &web_sys::CanvasRenderingContext2d,
    text: &str,
    x: f64, y: f64,
    color: &str,
    above: bool,
) {
    ctx.set_font("500 13px 'DM Sans',system-ui,sans-serif");
    let tw = ctx.measure_text(text).map(|m| m.width()).unwrap_or(80.0);
    let th = 13.0f64;
    let pad = 4.0;
    let draw_above = above && y - th - pad * 2.0 > 0.0;
    let (text_y, bg_y1, bg_y2) = if draw_above {
        (y - pad - 2.0, y - th - pad * 2.0, y)
    } else {
        (y + th + pad, y, y + th + pad * 2.0)
    };
    ctx.set_fill_style(&SHADOW.into());
    ctx.fill_rect(x - pad, bg_y1, tw + pad * 2.0, bg_y2 - bg_y1 + 2.0);
    ctx.set_fill_style(&color.into());
    let _ = ctx.fill_text(text, x, text_y);
}

#[component]
pub fn RecognizeView(pipeline: Rc<Pipeline>) -> impl IntoView {
    let video_ref = create_node_ref::<leptos::html::Video>();
    let canvas_ref = create_node_ref::<leptos::html::Canvas>();
    let (status, set_status) = create_signal(String::from("Starting camera..."));

    let pipeline_rc = pipeline.clone();
    create_effect(move |_| {
        let video_el = video_ref.get();
        let canvas_el = canvas_ref.get();
        if video_el.is_none() || canvas_el.is_none() { return; }
        let video = video_el.unwrap();
        let canvas = canvas_el.unwrap();

        let pipeline = pipeline_rc.clone();
        let set_status = set_status.clone();

        spawn_local(async move {
            let window = web_sys::window().unwrap();
            let media_devices = match window.navigator().media_devices() {
                Ok(md) => md,
                Err(_) => { set_status.set("Cannot access media devices.".into()); return; }
            };

            let constraints = web_sys::MediaStreamConstraints::new();
            constraints.set_video(&js_sys::Boolean::from(true).into());
            constraints.set_audio(&js_sys::Boolean::from(false).into());

            let stream_promise = match media_devices.get_user_media_with_constraints(&constraints) {
                Ok(p) => p,
                Err(_) => { set_status.set("Camera access denied.".into()); return; }
            };

            let stream_val = match wasm_bindgen_futures::JsFuture::from(stream_promise).await {
                Ok(v) => v,
                Err(_) => { set_status.set("Camera access denied.".into()); return; }
            };

            let stream = stream_val.dyn_into::<web_sys::MediaStream>().unwrap();
            video.set_src_object(Some(&stream));

            let _ = wasm_bindgen_futures::JsFuture::from(
                js_sys::Promise::new(&mut |resolve, _| {
                    let v = video.clone();
                    let c = canvas.clone();
                    let cb = wasm_bindgen::closure::Closure::once_into_js(move || {
                        c.set_width(v.video_width());
                        c.set_height(v.video_height());
                        resolve.call0(&wasm_bindgen::JsValue::UNDEFINED).unwrap();
                    });
                    video.set_onloadedmetadata(Some(cb.as_ref().unchecked_ref()));
                }),
            ).await;

            set_status.set(String::new());

            // Shared results between the two loops.
            let (results_read, results_write) = create_signal(Vec::<RecognitionResult>::new());

            // Render loop — runs at display refresh rate via requestAnimationFrame.
            {
                let canvas = canvas.clone();
                let video = video.clone();
                let perf = window.performance().unwrap();
                spawn_local(async move {
                    let mut fps_times: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
                    loop {
                        raf_promise().await;
                        if video.ready_state() < 2 { continue; }
                        let now = perf.now();
                        fps_times.push_back(now);
                        while fps_times.len() > 60 { fps_times.pop_front(); }
                        let fps = if fps_times.len() >= 2 {
                            (fps_times.len() - 1) as f64 / ((now - fps_times[0]) / 1000.0)
                        } else { 0.0 };
                        draw_overlay(&canvas, &results_read.get_untracked(), fps);
                    }
                });
            }

            // Inference loop — runs as fast as the model allows; updates results signal.
            loop {
                if video.ready_state() < 2 {
                    gloo_timers::future::TimeoutFuture::new(10).await;
                    continue;
                }
                match pipeline.recognize(&video, 0.4).await {
                    Ok(results) => {
                        set_status.set(if results.is_empty() {
                            "No face detected".into()
                        } else {
                            let matched: Vec<_> = results.iter()
                                .filter(|r| r.matched)
                                .map(|r| r.identity.as_str())
                                .collect();
                            if matched.is_empty() {
                                format!("{} face(s) not in profiles", results.len())
                            } else {
                                format!("Recognised: {}", matched.join(", "))
                            }
                        });
                        results_write.set(results);
                    }
                    Err(e) => {
                        set_status.set(format!("Error: {e}"));
                        gloo_timers::future::TimeoutFuture::new(500).await;
                    }
                }
                // Pause between inferences so the render loop and browser UI stay responsive.
                gloo_timers::future::TimeoutFuture::new(80).await;
            }
        });
    });

    view! {
        <div class="view-header">
            <h2>"Recognize"</h2>
            <p class="muted">"Live webcam recognition. Identities matched from your local profiles."</p>
        </div>
        <div class="video-wrap">
            <video node_ref=video_ref class="cam-video" autoplay playsinline muted />
            <canvas node_ref=canvas_ref class="cam-overlay" />
        </div>
        <div class="status-line">{move || status.get()}</div>
    }
}
