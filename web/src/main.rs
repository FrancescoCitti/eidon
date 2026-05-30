/// Eidon Web — fully client-side face recognition compiled to WebAssembly.
/// No server, no cloud, no data leaves the device.

mod app;
mod gallery;
mod inference;
mod pipeline;
mod types;

use leptos::*;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

use app::{enroll::EnrollView, gallery_view::GalleryView, recognize::RecognizeView};
use inference::{detector::Detector, embedder::Embedder};
use pipeline::Pipeline;
use gallery::gallery_clear;

const DET_MODEL: &str = "./models/buffalo_s/det_500m.onnx";
const REC_MODEL: &str = "./models/buffalo_s/w600k_mbf.onnx";

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

async fn load_pipeline() -> Result<Pipeline, String> {
    // Let ort-web fetch both models concurrently via URL — no manual byte transfer needed.
    let (detector, embedder) = futures::join!(
        Detector::from_url(DET_MODEL),
        Embedder::from_url(REC_MODEL),
    );
    Ok(Pipeline { detector: detector?, embedder: embedder? })
}

// ---------------------------------------------------------------------------
// App component
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
enum Tab { Recognize, Enroll, Profiles }

#[component]
fn App() -> impl IntoView {
    let (pipeline, set_pipeline) = create_signal(None::<Rc<Pipeline>>);
    let (error, set_error) = create_signal(None::<String>);
    let (loading_msg, set_loading_msg) = create_signal("Loading models…".to_string());
    let (active_tab, set_tab) = create_signal(Tab::Recognize);

    // Load models once on mount
    create_effect(move |_| {
        spawn_local(async move {
            set_loading_msg.set("Loading models…".into());
            match load_pipeline().await {
                Ok(p) => {
                    set_loading_msg.set("Ready".into());
                    set_pipeline.set(Some(Rc::new(p)));
                }
                Err(e) => set_error.set(Some(e)),
            }
        });
    });

    view! {
        {move || {
            if let Some(msg) = error.get() {
                return view! {
                    <div class="loading-screen">
                        <div class="loading-logo error">"!"</div>
                        <div class="loading-msg error">{msg}</div>
                        <div class="loading-hint">
                            "Ensure ONNX models are present at "
                            <code>"models/buffalo_s/det_500m.onnx"</code>
                            " and "
                            <code>"models/buffalo_s/w600k_mbf.onnx"</code>
                            "."
                        </div>
                    </div>
                }.into_view();
            }

            if pipeline.get().is_none() {
                return view! {
                    <div class="loading-screen">
                        <div class="loading-logo">"EIDON"</div>
                        <div class="loading-msg">{move || loading_msg.get()}</div>
                        <div class="loading-bar-wrap">
                            <div class="loading-bar" style="width:60%" />
                        </div>
                        <div class="loading-hint">
                            "Loading ONNX models (onnxruntime-web). First load may take a few seconds."
                        </div>
                    </div>
                }.into_view();
            }

            let p = pipeline.get().unwrap();

            view! {
                <header class="header">
                    <div class="logo">"EIDON Face Recognition"</div>
                    <nav class="tabs">
                        <button
                            class=move || if active_tab.get() == Tab::Recognize { "tab active" } else { "tab" }
                            on:click=move |_| set_tab.set(Tab::Recognize)
                        >"Recognize"</button>
                        <button
                            class=move || if active_tab.get() == Tab::Enroll { "tab active" } else { "tab" }
                            on:click=move |_| set_tab.set(Tab::Enroll)
                        >"Enroll"</button>
                        <button
                            class=move || if active_tab.get() == Tab::Profiles { "tab active" } else { "tab" }
                            on:click=move |_| set_tab.set(Tab::Profiles)
                        >"Profiles"</button>
                    </nav>
                </header>
                <main class="main">
                    {move || match active_tab.get() {
                        Tab::Recognize => view! {
                            <RecognizeView pipeline=p.clone() />
                        }.into_view(),
                        Tab::Enroll => view! {
                            <EnrollView
                                pipeline=p.clone()
                                on_done=Callback::new(move |()| set_tab.set(Tab::Recognize))
                            />
                        }.into_view(),
                        Tab::Profiles => view! {
                            <GalleryView />
                        }.into_view(),
                    }}
                </main>
                <footer class="app-footer">
                    <div class="footer-main">
                        <div class="footer-text">
                            <p>
                                "All image processing and face recognition runs entirely within your browser — "
                                "no photographs, video frames, or biometric data are ever transmitted to any server or third party. "
                                "Enrolled face embeddings are stored exclusively in your browser's IndexedDB. "
                                "They persist across sessions on this device and are never shared. "
                                "Closing this page does not delete your data; use the button below to erase everything permanently."
                            </p>
                            <p class="footer-legal">
                                "This application processes biometric data locally in accordance with the principle of privacy by design "
                                "and data minimisation as defined in the EU General Data Protection Regulation "
                                "(Regulation (EU) 2016/679, Art. 5(1)(c) and Art. 25). "
                                "No personal data leaves your device and no third-party data processing occurs."
                            </p>
                        </div>
                        <button
                            class="btn btn-danger btn-sm footer-clear-btn"
                            on:click=move |_| {
                                spawn_local(async move {
                                    let _ = gallery_clear().await;
                                });
                            }
                        >
                            "Clear all data"
                        </button>
                    </div>
                </footer>
            }.into_view()
        }}
    }
}

fn main() {
    leptos::mount_to(
        web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("app")
            .unwrap()
            .dyn_into()
            .unwrap(),
        App,
    );
}
