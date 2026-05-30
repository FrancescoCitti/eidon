/// Gallery management view.

use leptos::*;
use wasm_bindgen_futures::spawn_local;

use crate::gallery::{gallery_clear, gallery_get_all, gallery_list, gallery_remove};

#[component]
pub fn GalleryView() -> impl IntoView {
    let (identities, set_identities) = create_signal(Vec::<(String, usize)>::new());
    let (loading, set_loading) = create_signal(true);

    let refresh = {
        let set_identities = set_identities.clone();
        let set_loading = set_loading.clone();
        move || {
            let set_identities = set_identities.clone();
            let set_loading = set_loading.clone();
            spawn_local(async move {
                match gallery_get_all().await {
                    Ok(map) => {
                        let mut list: Vec<(String, usize)> = map
                            .into_iter()
                            .map(|(k, v)| (k, v.len()))
                            .collect();
                        list.sort_by(|a, b| a.0.cmp(&b.0));
                        set_identities.set(list);
                    }
                    Err(_) => set_identities.set(Vec::new()),
                }
                set_loading.set(false);
            });
        }
    };

    // Load on mount
    {
        let refresh = refresh.clone();
        create_effect(move |_| refresh());
    }

    let on_remove = {
        let refresh = refresh.clone();
        move |name: String| {
            let refresh = refresh.clone();
            spawn_local(async move {
                let _ = gallery_remove(&name).await;
                refresh();
            });
        }
    };

    let on_clear = {
        let refresh = refresh.clone();
        move || {
            let refresh = refresh.clone();
            spawn_local(async move {
                let _ = gallery_clear().await;
                refresh();
            });
        }
    };

    view! {
        <div class="view-header">
            <h2>"Gallery"</h2>
            <p class="muted">
                "Embeddings are stored in your browser's IndexedDB. They never leave your device."
            </p>
        </div>
        {move || if loading.get() {
            view! { <p class="muted">"Loading…"</p> }.into_view()
        } else {
            let ids = identities.get();
            if ids.is_empty() {
                view! {
                    <p class="muted empty-msg">
                        "No identities enrolled yet. Go to Enroll to add someone."
                    </p>
                }.into_view()
            } else {
                let on_remove = on_remove.clone();
                let on_clear = on_clear.clone();
                view! {
                    <div class="gallery-list">
                        {ids.iter().map(|(name, count)| {
                            let name = name.clone();
                            let on_remove = on_remove.clone();
                            view! {
                                <div class="identity-row">
                                    <div>
                                        <span class="identity-name">{name.clone()}</span>
                                        <span class="identity-count">
                                            {*count}" embedding"{if *count != 1 { "s" } else { "" }}
                                        </span>
                                    </div>
                                    <button
                                        class="btn btn-danger btn-sm"
                                        on:click={
                                            let name = name.clone();
                                            let on_remove = on_remove.clone();
                                            move |_| on_remove(name.clone())
                                        }
                                    >"Remove"</button>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                    <button
                        class="btn btn-danger"
                        on:click=move |_| on_clear()
                    >"Clear all"</button>
                }.into_view()
            }
        }}
    }
}
