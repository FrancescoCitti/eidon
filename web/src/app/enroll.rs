use leptos::*;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

use crate::pipeline::Pipeline;
use crate::types::Detection;

const POSES: &[(&str, &str)] = &[
    ("Look straight at the camera", "😐"),
    ("Turn your head RIGHT",        "↪"),
    ("Turn your head LEFT",         "↩"),
    ("Tilt your head RIGHT",        "↗"),
    ("Tilt your head LEFT",         "↖"),
];
const DWELL_MS: f64 = 1500.0;
const CAPTURES_PER_POSE: usize = 2;
const MAX_UPLOAD_IMAGES: u32 = 10;

struct PoseMetrics { yaw: f32, roll: f32 }

fn compute_metrics(det: &Detection) -> PoseMetrics {
    let lx = det.landmarks[0]; let ly = det.landmarks[1];
    let rx = det.landmarks[2]; let ry = det.landmarks[3];
    let nx = det.landmarks[4];
    let eye_mid_x = (lx + rx) / 2.0;
    let eye_dist = ((rx - lx).powi(2) + (ry - ly).powi(2)).sqrt().max(1.0);
    PoseMetrics {
        yaw:  (nx - eye_mid_x) / eye_dist,
        roll: (ry - ly).atan2(rx - lx).to_degrees(),
    }
}

fn check_pose(idx: usize, m: &PoseMetrics) -> Option<&'static str> {
    match idx {
        0 => {
            if m.yaw.abs() > 0.22   { Some("Face the camera directly") }
            else if m.roll.abs() > 14.0 { Some("Keep your head level") }
            else { None }
        }
        1 => { if m.yaw > -0.18 { Some("Turn further to the right") } else { None } }
        2 => { if m.yaw < 0.18  { Some("Turn further to the left")  } else { None } }
        3 => { if m.roll < 11.0 { Some("Tilt further to the right") } else { None } }
        4 => { if m.roll > -11.0{ Some("Tilt further to the left")  } else { None } }
        _ => None,
    }
}

#[derive(Clone, PartialEq)]
enum EnrollState {
    NameEntry,
    MethodChoice { name: String },
    Camera { name: String },
    Upload { name: String },
    Done { name: String, total: usize },
}

#[component]
pub fn EnrollView(pipeline: Rc<Pipeline>, on_done: Callback<()>) -> impl IntoView {
    let (state, set_state) = create_signal(EnrollState::NameEntry);
    let (name_input, set_name_input) = create_signal(String::new());

    let start = {
        let set_state = set_state.clone();
        let name_input = name_input.clone();
        move || {
            let name = name_input.get_untracked().trim().to_owned();
            if !name.is_empty() {
                set_state.set(EnrollState::MethodChoice { name });
            }
        }
    };

    view! {
        {move || match state.get() {
            EnrollState::NameEntry => view! {
                <div class="view-header">
                    <h2>"Enroll Identity"</h2>
                    <p class="muted">"Add a new person to your local profiles."</p>
                </div>
                <div class="enroll-name-row">
                    <input
                        class="text-input"
                        type="text"
                        placeholder="Identity name (e.g. Alice)"
                        prop:value=move || name_input.get()
                        on:input=move |e| set_name_input.set(event_target_value(&e))
                        on:keydown=move |e| { if e.key() == "Enter" { start(); } }
                    />
                    <button class="btn btn-primary" on:click=move |_| start()>"Next"</button>
                </div>
            }.into_view(),

            EnrollState::MethodChoice { name } => {
                let n1 = name.clone(); let n2 = name.clone();
                let ss1 = set_state.clone(); let ss2 = set_state.clone();
                view! {
                    <div class="view-header">
                        <h2>"Choose Method"</h2>
                        <p class="muted">"How would you like to enroll "{name.clone()}"?"</p>
                    </div>
                    <div class="method-choice">
                        <button class="method-card" on:click=move |_| ss1.set(EnrollState::Camera { name: n1.clone() })>
                            <span class="method-icon">"📷"</span>
                            <span class="method-title">"Use Camera"</span>
                            <span class="method-desc">"5 guided poses captured live"</span>
                        </button>
                        <button class="method-card" on:click=move |_| ss2.set(EnrollState::Upload { name: n2.clone() })>
                            <span class="method-icon">"🖼"</span>
                            <span class="method-title">"Upload Photos"</span>
                            <span class="method-desc">"Select up to 10 images from disk"</span>
                        </button>
                    </div>
                }.into_view()
            },

            EnrollState::Camera { name } => view! {
                <SessionView
                    pipeline=pipeline.clone()
                    identity=name.clone()
                    on_done=Callback::new({
                        let set_state = set_state.clone();
                        move |(n, total): (String, usize)| {
                            set_state.set(EnrollState::Done { name: n, total });
                        }
                    })
                />
            }.into_view(),

            EnrollState::Upload { name } => view! {
                <UploadView
                    pipeline=pipeline.clone()
                    identity=name.clone()
                    on_done=Callback::new({
                        let set_state = set_state.clone();
                        move |(n, total): (String, usize)| {
                            set_state.set(EnrollState::Done { name: n, total });
                        }
                    })
                    on_cancel=Callback::new({
                        let set_state = set_state.clone();
                        let nm = name.clone();
                        move |()| set_state.set(EnrollState::MethodChoice { name: nm.clone() })
                    })
                />
            }.into_view(),

            EnrollState::Done { name, total } => view! {
                <div class="enroll-done">
                    <div class="done-icon">"✓"</div>
                    <h3>{name.clone()}" enrolled"</h3>
                    <p>{total}" embedding(s) saved to your local profiles."</p>
                    <button class="btn btn-primary" on:click=move |_| on_done.call(())>
                        "Back to Recognize"
                    </button>
                </div>
            }.into_view(),
        }}
    }
}

#[component]
fn UploadView(
    pipeline: Rc<Pipeline>,
    identity: String,
    on_done: Callback<(String, usize)>,
    on_cancel: Callback<()>,
) -> impl IntoView {
    let (files, set_files) = create_signal(Vec::<web_sys::File>::new());
    let (processing, set_processing) = create_signal(false);
    let (progress, set_progress) = create_signal(0usize);
    let (status, set_status) = create_signal(String::new());

    let on_change = move |e: web_sys::Event| {
        let input = e.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap();
        if let Some(file_list) = input.files() {
            let count = file_list.length().min(MAX_UPLOAD_IMAGES);
            let v: Vec<web_sys::File> = (0..count).filter_map(|i| file_list.item(i)).collect();
            set_files.set(v);
        }
    };

    let pipeline_rc = pipeline.clone();
    let identity_clone = identity.clone();

    let process = move || {
        let files = files.get_untracked();
        if files.is_empty() { return; }
        let pipeline = pipeline_rc.clone();
        let identity = identity_clone.clone();

        spawn_local(async move {
            set_processing.set(true);
            let mut enrolled = 0usize;

            for (i, file) in files.iter().enumerate() {
                set_progress.set(i + 1);
                set_status.set(format!("Processing image {} of {}...", i + 1, files.len()));

                match load_and_enroll(&pipeline, file, &identity).await {
                    Ok(true)  => enrolled += 1,
                    Ok(false) => {},
                    Err(e)    => set_status.set(format!("Image {}: {e}", i + 1)),
                }
            }

            set_processing.set(false);
            if enrolled > 0 {
                on_done.call((identity.clone(), enrolled));
            } else {
                set_status.set("No faces detected in the uploaded images. Try different photos.".into());
            }
        });
    };

    view! {
        <div class="view-header">
            <h2>"Upload Photos"</h2>
            <p class="muted">"Select up to 10 clear, well-lit photos of the person's face."</p>
        </div>
        <div class="upload-area">
            <input
                type="file"
                accept="image/*"
                multiple
                class="file-input"
                disabled=move || processing.get()
                on:change=on_change
            />
            {move || {
                let f = files.get();
                if f.is_empty() {
                    view! { <span class="upload-hint">"No files selected"</span> }.into_view()
                } else {
                    view! {
                        <ul class="file-list">
                            {f.iter().map(|file| {
                                let name = file.name();
                                view! { <li class="file-item">{name}</li> }
                            }).collect_view()}
                        </ul>
                    }.into_view()
                }
            }}
        </div>
        {move || if processing.get() {
            let pct = progress.get() as f64 / files.get().len().max(1) as f64 * 100.0;
            view! {
                <div class="upload-progress">
                    <div class="loading-bar-wrap" style="width:100%">
                        <div class="loading-bar" style=move || format!("width:{pct:.0}%") />
                    </div>
                </div>
            }.into_view()
        } else {
            view! { <span /> }.into_view()
        }}
        <div class="controls row">
            <button
                class="btn btn-primary"
                disabled=move || processing.get() || files.get().is_empty()
                on:click=move |_| process()
            >
                {move || {
                    let n = files.get().len();
                    if n == 0 { "Process".to_string() } else { format!("Process {} image(s)", n) }
                }}
            </button>
            <button
                class="btn btn-ghost"
                disabled=move || processing.get()
                on:click=move |_| on_cancel.call(())
            >"Cancel"</button>
        </div>
        <div class="status-line">{move || status.get()}</div>
    }
}

async fn load_and_enroll(
    pipeline: &Pipeline,
    file: &web_sys::File,
    identity: &str,
) -> Result<bool, String> {
    let url = web_sys::Url::create_object_url_with_blob(file)
        .map_err(|e| format!("{e:?}"))?;

    let img = web_sys::HtmlImageElement::new()
        .map_err(|e| format!("{e:?}"))?;
    img.set_src(&url);

    wasm_bindgen_futures::JsFuture::from(
        js_sys::Promise::new(&mut |resolve, reject| {
            let ok = wasm_bindgen::closure::Closure::once_into_js(move || {
                resolve.call0(&wasm_bindgen::JsValue::UNDEFINED).unwrap();
            });
            let err = wasm_bindgen::closure::Closure::once_into_js(move || {
                reject.call0(&wasm_bindgen::JsValue::UNDEFINED).unwrap();
            });
            img.set_onload(Some(ok.as_ref().unchecked_ref()));
            img.set_onerror(Some(err.as_ref().unchecked_ref()));
        }),
    ).await.map_err(|_| "Failed to load image".to_string())?;

    let result = pipeline.enroll_image(&img, identity).await;
    web_sys::Url::revoke_object_url(&url).ok();
    result
}

#[component]
fn SessionView(
    pipeline: Rc<Pipeline>,
    identity: String,
    on_done: Callback<(String, usize)>,
) -> impl IntoView {
    let video_ref = create_node_ref::<leptos::html::Video>();
    let (pose_idx, set_pose_idx) = create_signal(0usize);
    let (_cap_this_pose, set_cap_this_pose) = create_signal(0usize);
    let (total_captures, set_total_captures) = create_signal(0usize);
    let (dwell_pct, set_dwell_pct) = create_signal(0.0f64);
    let (status, set_status) = create_signal(String::from("Starting camera..."));
    let (camera_ready, set_camera_ready) = create_signal(false);
    let (capture_trigger, set_capture_trigger) = create_signal(false);

    let identity_clone = identity.clone();
    let pipeline_rc = pipeline.clone();

    create_effect(move |_| {
        let Some(video) = video_ref.get() else { return };
        let set_camera_ready = set_camera_ready.clone();
        let set_status = set_status.clone();

        spawn_local(async move {
            let window = web_sys::window().unwrap();
            let constraints = web_sys::MediaStreamConstraints::new();
            constraints.set_video(&js_sys::Boolean::from(true).into());
            constraints.set_audio(&js_sys::Boolean::from(false).into());

            match window.navigator().media_devices()
                .and_then(|md| md.get_user_media_with_constraints(&constraints))
            {
                Ok(p) => {
                    if let Ok(v) = wasm_bindgen_futures::JsFuture::from(p).await {
                        let stream = v.dyn_into::<web_sys::MediaStream>().unwrap();
                        video.set_src_object(Some(&stream));
                        let _ = wasm_bindgen_futures::JsFuture::from(
                            js_sys::Promise::new(&mut |resolve, _| {
                                let cb = wasm_bindgen::closure::Closure::once_into_js(move || {
                                    resolve.call0(&wasm_bindgen::JsValue::UNDEFINED).unwrap();
                                });
                                video.set_onloadedmetadata(Some(cb.as_ref().unchecked_ref()));
                            }),
                        ).await;
                        set_status.set(format!("Pose 1 of {}", POSES.len()));
                        set_camera_ready.set(true);
                    }
                }
                Err(_) => set_status.set("Camera access denied.".into()),
            }
        });
    });

    {
        let pipeline_rc = pipeline_rc.clone();
        let identity = identity_clone.clone();

        create_effect(move |_| {
            if !camera_ready.get() { return; }
            let Some(video_el) = video_ref.get() else { return };
            let pipeline = pipeline_rc.clone();
            let identity = identity.clone();

            spawn_local(async move {
                let mut face_first_seen: Option<f64> = None;
                let perf = web_sys::window().unwrap().performance().unwrap();

                loop {
                    gloo_timers::future::TimeoutFuture::new(33).await;

                    if capture_trigger.get_untracked() {
                        set_capture_trigger.set(false);
                        do_capture(&pipeline, &video_el, &identity,
                            set_pose_idx, set_cap_this_pose, set_total_captures,
                            set_status.clone(), on_done.clone()).await;
                        face_first_seen = None;
                        set_dwell_pct.set(0.0);
                        continue;
                    }

                    let detections = pipeline.detector.detect(&video_el, 0.6, 0.4).await
                        .unwrap_or_default();

                    if detections.is_empty() {
                        face_first_seen = None;
                        set_dwell_pct.set(0.0);
                        set_status.set("No face detected.".into());
                        continue;
                    }

                    let det = detections.iter()
                        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                        .unwrap();

                    let metrics = compute_metrics(det);
                    let current_pose = pose_idx.get_untracked();

                    if let Some(hint) = check_pose(current_pose, &metrics) {
                        face_first_seen = None;
                        set_dwell_pct.set(0.0);
                        set_status.set(hint.to_owned());
                        continue;
                    }

                    let now = perf.now();
                    let first = face_first_seen.get_or_insert(now);
                    let elapsed = now - *first;
                    set_dwell_pct.set((elapsed / DWELL_MS * 100.0).min(100.0));
                    set_status.set(format!("Pose {} of {} — hold steady", current_pose + 1, POSES.len()));

                    if elapsed >= DWELL_MS {
                        face_first_seen = None;
                        set_dwell_pct.set(0.0);
                        do_capture(&pipeline, &video_el, &identity,
                            set_pose_idx, set_cap_this_pose, set_total_captures,
                            set_status.clone(), on_done.clone()).await;
                    }
                }
            });
        });
    }

    let pose_name = move || { let i = pose_idx.get(); if i < POSES.len() { POSES[i].0 } else { "" } };
    let pose_icon = move || { let i = pose_idx.get(); if i < POSES.len() { POSES[i].1 } else { "" } };

    view! {
        <div class="pose-dots">
            {(0..POSES.len()).map(|i| {
                let cls = move || {
                    let p = pose_idx.get();
                    if i < p { "dot done" } else if i == p { "dot active" } else { "dot" }
                };
                view! { <span class=cls /> }
            }).collect_view()}
        </div>
        <div class="pose-instruction">
            <span class="pose-icon">{pose_icon}</span>
            {pose_name}
        </div>
        <div class="video-wrap">
            <video node_ref=video_ref class="cam-video" autoplay playsinline muted />
            <div class="dwell-bar-wrap">
                <div class="dwell-bar" style=move || format!("width:{}%", dwell_pct.get()) />
            </div>
        </div>
        <div class="controls row">
            <button class="btn btn-accent" on:click=move |_| set_capture_trigger.set(true)>
                "Capture now"
            </button>
            <button
                class="btn btn-ghost"
                on:click=move |_| on_done.call((identity_clone.clone(), total_captures.get_untracked()))
            >"Cancel"</button>
        </div>
        <div class="status-line">{move || status.get()}</div>
    }
}

async fn do_capture(
    pipeline: &Pipeline,
    video: &web_sys::HtmlVideoElement,
    identity: &str,
    set_pose_idx: WriteSignal<usize>,
    set_cap_this_pose: WriteSignal<usize>,
    set_total_captures: WriteSignal<usize>,
    set_status: WriteSignal<String>,
    on_done: Callback<(String, usize)>,
) {
    match pipeline.enroll(video, identity).await {
        Ok(true) => {
            let total = set_total_captures.try_update(|n| { *n += 1; *n }).unwrap_or(0);
            let advance = set_cap_this_pose
                .try_update(|n| { *n += 1; *n >= CAPTURES_PER_POSE })
                .unwrap_or(false);
            if advance { set_cap_this_pose.set(0); }
            set_status.set(format!("Captured ({total} total)"));
            if advance {
                let finished = set_pose_idx
                    .try_update(|p| { *p += 1; *p >= POSES.len() })
                    .unwrap_or(true);
                if finished { on_done.call((identity.to_owned(), total)); }
            }
        }
        Ok(false) => set_status.set("No face detected. Position your face in the frame.".into()),
        Err(e) => set_status.set(format!("Error: {e}")),
    }
}
