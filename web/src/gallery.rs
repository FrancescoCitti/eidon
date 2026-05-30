/// IndexedDB gallery backed by `rexie`.
///
/// Each record stores one 512-dim embedding for one identity.
/// All face data stays on the user's device — nothing is sent to any server.

use rexie::{ObjectStore, Rexie, TransactionMode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const DB_NAME: &str = "eidon-gallery";
const DB_VERSION: u32 = 1;
const STORE: &str = "embeddings";

#[derive(Serialize, Deserialize, Clone)]
struct Record {
    identity: String,
    embedding: Vec<f32>,
}

async fn open_db() -> Result<Rexie, rexie::Error> {
    Rexie::builder(DB_NAME)
        .version(DB_VERSION)
        .add_object_store(ObjectStore::new(STORE).auto_increment(true))
        .build()
        .await
}

/// Store one L2-normalised (512,) embedding for `identity`.
pub async fn gallery_add(identity: &str, embedding: Vec<f32>) -> Result<(), String> {
    let db = open_db().await.map_err(|e| e.to_string())?;
    let tx = db
        .transaction(&[STORE], TransactionMode::ReadWrite)
        .map_err(|e| e.to_string())?;
    let store = tx.store(STORE).map_err(|e| e.to_string())?;
    let record = Record { identity: identity.to_owned(), embedding };
    let val = serde_wasm_bindgen::to_value(&record).map_err(|e| e.to_string())?;
    store.add(&val, None).await.map_err(|e| e.to_string())?;
    tx.done().await.map_err(|e| e.to_string())?;
    Ok(())
}

/// Return all embeddings grouped by identity.
pub async fn gallery_get_all() -> Result<HashMap<String, Vec<Vec<f32>>>, String> {
    let db = open_db().await.map_err(|e| e.to_string())?;
    let tx = db
        .transaction(&[STORE], TransactionMode::ReadOnly)
        .map_err(|e| e.to_string())?;
    let store = tx.store(STORE).map_err(|e| e.to_string())?;
    let all = store.get_all(None, None, None, None).await.map_err(|e| e.to_string())?;

    let mut map: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for (_, val) in all {
        let rec: Record = serde_wasm_bindgen::from_value(val).map_err(|e| e.to_string())?;
        map.entry(rec.identity).or_default().push(rec.embedding);
    }
    Ok(map)
}

/// Sorted list of enrolled identity names.
pub async fn gallery_list() -> Result<Vec<String>, String> {
    let map = gallery_get_all().await?;
    let mut names: Vec<String> = map.into_keys().collect();
    names.sort();
    Ok(names)
}

/// Number of embeddings stored for `identity`.
pub async fn gallery_count(identity: &str) -> Result<usize, String> {
    let map = gallery_get_all().await?;
    Ok(map.get(identity).map_or(0, |v| v.len()))
}

/// Remove all embeddings for `identity`.
pub async fn gallery_remove(identity: &str) -> Result<(), String> {
    let db = open_db().await.map_err(|e| e.to_string())?;
    let tx = db
        .transaction(&[STORE], TransactionMode::ReadWrite)
        .map_err(|e| e.to_string())?;
    let store = tx.store(STORE).map_err(|e| e.to_string())?;

    // Collect keys + records, delete matching ones
    let all = store.get_all(None, None, None, None).await.map_err(|e| e.to_string())?;
    for (key, val) in all {
        let rec: Record = match serde_wasm_bindgen::from_value(val) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if rec.identity == identity {
            store.delete(&key).await.map_err(|e| e.to_string())?;
        }
    }
    tx.done().await.map_err(|e| e.to_string())?;
    Ok(())
}

/// Erase all records in the gallery.
pub async fn gallery_clear() -> Result<(), String> {
    let db = open_db().await.map_err(|e| e.to_string())?;
    let tx = db
        .transaction(&[STORE], TransactionMode::ReadWrite)
        .map_err(|e| e.to_string())?;
    tx.store(STORE).map_err(|e| e.to_string())?.clear().await.map_err(|e| e.to_string())?;
    tx.done().await.map_err(|e| e.to_string())?;
    Ok(())
}
