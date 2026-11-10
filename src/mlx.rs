//! The optional MLX embedding backend: a self-owned Python sidecar process and
//! its one-time setup command. MIT. The sidecar (`sidecar/mlx_serve.py`) speaks
//! line-delimited JSON over stdin/stdout and runs ModernBERT on mlx (MIT) with
//! a transformers (Apache-2.0) tokenizer; the GPL-3.0 mlx-embeddings package is
//! deliberately NOT used anywhere, and the sidecar imports neither torch nor
//! any networking library — downloads belong to `setup-mlx` only.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::ModelSpec;

/// The single actionable recovery step for every MLX availability failure.
pub const SETUP_HINT: &str =
    "run `rusty-brain setup-mlx` to install the MLX sidecar venv and model";

pub const MLX_SIDECAR_SCRIPT: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/sidecar/mlx_serve.py"));

/// Pinned, license-checked Python dependencies (mlx: MIT, transformers:
/// Apache-2.0). mlx-embeddings (GPL-3.0) is deliberately not installed.
const VENV_PINS: &[&str] = &["mlx==0.32.4", "transformers==5.22.2"];
/// Files fetched from the model's Hugging Face repo; `model.safetensors` is
/// additionally size-verified below.
const MODEL_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
];
const VENV_DIR: &str = "mlx-venv";

fn mlx_spec() -> &'static ModelSpec {
    crate::config::MODELS
        .iter()
        .find(|m| m.backend == crate::config::Backend::Mlx)
        .expect("exactly one MLX arm is registered")
}

fn run(command: &mut Command, what: &str) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("spawning {what}"))?;
    ensure!(status.success(), "{what} failed with {status}");
    Ok(())
}

/// The server-reported byte size of a model file, from `curl -I`. `None` means
/// "unknown" (header fetch failed): the caller then falls back to non-empty.
fn remote_size(url: &str) -> Option<u64> {
    let output = Command::new("curl")
        .args(["-fsIL", "--retry", "2", url])
        .env("HF_HUB_DISABLE_XET", "1")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let headers = String::from_utf8_lossy(&output.stdout);
    headers
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name.trim().eq_ignore_ascii_case("content-length"))
                .then(|| value.trim().parse().ok())
                .flatten()
        })
        .next_back()
}

/// `rusty-brain setup-mlx`: write the embedded sidecar, create the pinned
/// Python venv, and download/verify the model artifact — every step idempotent,
/// rerun = already-present + verified.
pub fn setup(home: &Path) -> Result<()> {
    let spec = mlx_spec();
    crate::config::ensure_private_dir(home)?;

    println!("MLX setup for {} ({}, Apache-2.0)", spec.id, spec.source);

    // 1. The sidecar script, from the copy embedded in this binary.
    let sidecar_path = home.join("sidecar").join("mlx_serve.py");
    crate::config::ensure_private_dir(sidecar_path.parent().expect("sidecar parent"))?;
    if sidecar_path.exists()
        && std::fs::read(&sidecar_path).is_ok_and(|bytes| bytes == MLX_SIDECAR_SCRIPT.as_bytes())
    {
        println!("sidecar script present: {}", sidecar_path.display());
    } else {
        std::fs::write(&sidecar_path, MLX_SIDECAR_SCRIPT)
            .with_context(|| format!("writing {}", sidecar_path.display()))?;
        crate::config::set_private_file(&sidecar_path)?;
        println!("sidecar script written: {}", sidecar_path.display());
    }

    // 2. uv, the only external tool this step needs.
    let uv_present = Command::new("uv")
        .arg("--version")
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    ensure!(
        uv_present,
        "uv is required to install the MLX sidecar venv; install uv (https://docs.astral.sh/uv/)"
    );

    // 3. The pinned venv.
    let venv = home.join(VENV_DIR);
    let python = venv.join("bin").join("python");
    if !python.exists() {
        run(
            Command::new("uv")
                .arg("venv")
                .arg(&venv)
                .arg("--python")
                .arg("3.12")
                .env("HF_HUB_DISABLE_XET", "1"),
            "creating the MLX venv (uv venv)",
        )?;
    } else {
        println!("venv present: {}", venv.display());
    }

    // 4. Pinned dependencies. uv verifies the existing set quickly, so this is
    // safe to rerun.
    let mut install = Command::new("uv");
    install
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .args(VENV_PINS)
        .env("HF_HUB_DISABLE_XET", "1");
    println!("installing pinned venv packages: {}", VENV_PINS.join(" "));
    run(
        &mut install,
        "installing the pinned MLX venv packages (uv pip install)",
    )?;

    // 5. The model artifact, resumable per file. "Fresh" means the local size
    // matches the server's content-length (exact, when the server answers
    // HEAD requests), so a truncated earlier attempt always redownloads.
    let model_dir = home.join("models").join(spec.id);
    crate::config::ensure_private_dir(&model_dir)?;
    let base = format!("https://huggingface.co/{}/resolve/main/", spec.source);
    let mut expected_sizes: Vec<(&'static str, Option<u64>)> = Vec::new();
    for file in MODEL_FILES {
        let destination = model_dir.join(file);
        let url = format!("{base}{file}");
        let expected = remote_size(&url);
        let fresh = expected.is_some_and(|expected| {
            std::fs::metadata(&destination)
                .map(|meta| meta.len() == expected)
                .unwrap_or(false)
        });
        if fresh {
            expected_sizes.push((file, expected));
            continue;
        }
        println!("downloading {url}");
        run(
            Command::new("curl")
                .args(["-fL", "--retry", "5", "-C", "-"])
                .arg(&url)
                .arg("-o")
                .arg(&destination)
                .env("HF_HUB_DISABLE_XET", "1"),
            "downloading the model artifact (curl)",
        )?;
        expected_sizes.push((file, expected));
    }

    // 6. Verify: every file's exact reported size when known, plus a real
    // import of the pinned versions.
    for (file, expected) in expected_sizes {
        let path = model_dir.join(file);
        let size = std::fs::metadata(&path)
            .with_context(|| format!("verifying {}", path.display()))?
            .len();
        match expected {
            Some(expected) => ensure!(
                size == expected,
                "{file} is incomplete ({size} of {expected} bytes); rerun `rusty-brain setup-mlx` to resume the download"
            ),
            None => ensure!(
                size > 0,
                "{file} is empty; rerun `rusty-brain setup-mlx` to resume"
            ),
        }
    }
    run(
        Command::new(&python)
            .arg("-I")
            .arg("-c")
            .arg("import mlx.core as mx, transformers; print('mlx', mx.__version__)")
            .env("HF_HUB_DISABLE_XET", "1"),
        "verifying the MLX venv imports",
    )?;
    println!(
        "MLX setup complete: {} ready in {}",
        spec.id,
        home.display()
    );
    Ok(())
}

/// Document/query embeddings for the MLX arm, computed by the sidecar child
/// process. Exposes the same `embed`/`embed_query` surface as the fastembed
/// `Embedder`; document and query prefixes are applied here, per the spec.
/// All mutation happens through `&mut self`, so the brain's existing embedder
/// locks serialize every child request.
pub struct MlxEmbedder {
    spec: &'static ModelSpec,
    model_dir: PathBuf,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: AtomicU64,
}

impl MlxEmbedder {
    /// Spawn the sidecar for one model and perform the load handshake.
    pub fn new(models_dir: &Path, spec: &'static ModelSpec) -> Result<Self> {
        let home = models_dir
            .parent()
            .context("models dir must live inside the rusty-brain home")?;
        let script = home.join("sidecar").join("mlx_serve.py");
        let model_dir = models_dir.join(spec.id);
        // The override names a command (PATH name or absolute path); the default
        // is the setup-created venv python, which must exist on disk.
        let (python, python_ready) = match std::env::var("RUSTY_BRAIN_MLX_PYTHON") {
            Ok(custom) if !custom.trim().is_empty() => (PathBuf::from(custom.trim()), true),
            _ => {
                let default_python = home.join(VENV_DIR).join("bin").join("python");
                let ready = default_python.is_file();
                (default_python, ready)
            }
        };
        let weights_ready = std::fs::metadata(model_dir.join("model.safetensors"))
            .map(|meta| meta.len() > 0)
            .unwrap_or(false);
        if !python_ready || !script.is_file() || !weights_ready {
            bail!(
                "MLX backend is not ready for model {}: {SETUP_HINT}",
                spec.id
            );
        }
        let mut child = Command::new(&python)
            .arg("-I")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("HF_HUB_DISABLE_XET", "1")
            .spawn()
            .with_context(|| format!("spawning the MLX sidecar with {}", python.display()))?;
        // The stdin/stdout handles are detached from the Child handle: dropping
        // the embedder closes stdin, the sidecar exits on EOF, and the child
        // itself dies with the brain process.
        let stdin = child.stdin.take().context("sidecar stdin is piped")?;
        let stdout = child.stdout.take().context("sidecar stdout is piped")?;
        let mut embedder = Self {
            spec,
            model_dir,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: AtomicU64::new(0),
        };
        // The load failure bails with a fully actionable message itself; an
        // anyhow context wrapper would hide it from top-level diagnostics.
        embedder.load()?;
        Ok(embedder)
    }

    fn load(&mut self) -> Result<()> {
        let payload = json!({"op": "load", "artifact": self.model_dir, "model": self.spec.id});
        writeln!(self.stdin, "{payload}").context("sending load request to the MLX sidecar")?;
        self.stdin.flush().context("flushing load request")?;
        let reply = self.read_reply("load")?;
        let error = reply.get("error").and_then(Value::as_str);
        match error {
            Some(error) => bail!(
                "the MLX sidecar failed to load {}: {error}; reinstall the backend with `rusty-brain setup-mlx`",
                self.spec.id
            ),
            None => {
                ensure!(
                    reply.get("ok").and_then(Value::as_bool).unwrap_or(false),
                    "unexpected MLX sidecar load reply: {reply}"
                );
            }
        }
        Ok(())
    }

    fn read_reply(&mut self, op: &str) -> Result<Value> {
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .context("reading a reply from the MLX sidecar")?;
        if bytes == 0 {
            bail!("the MLX sidecar exited before replying to {op}; {SETUP_HINT}");
        }
        serde_json::from_str(&line).with_context(|| format!("parsing the MLX sidecar {op} reply"))
    }

    fn embed_prefixed(&mut self, prefixed: &str) -> Result<Vec<f32>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let payload = json!({"op": "embed", "id": id, "text": prefixed});
        writeln!(self.stdin, "{payload}").context("sending an embed request to the MLX sidecar")?;
        self.stdin.flush().context("flushing an embed request")?;
        let reply = self.read_reply("embed")?;
        if let Some(error) = reply.get("error").and_then(Value::as_str) {
            bail!("MLX sidecar embed failed: {error}");
        }
        let embedding = reply
            .get("embedding")
            .and_then(Value::as_array)
            .context("MLX sidecar reply has no embedding")?;
        let mut vector = Vec::with_capacity(embedding.len());
        for value in embedding {
            vector.push(
                value
                    .as_f64()
                    .with_context(|| format!("embedding value {value} is not a number"))?
                    as f32,
            );
        }
        ensure!(
            vector.len() == self.spec.dim as usize,
            "MLX sidecar returned {} values, expected {}",
            vector.len(),
            self.spec.dim
        );
        Ok(vector)
    }

    /// Document embedding (memories): applies the spec's document prefix.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_prefixed(&format!("{}{text}", self.spec.doc_prefix))
    }

    /// Query embedding: applies the spec's query prefix, never the doc one.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        if self.spec.query_prefix.is_empty() {
            self.embed_prefixed(text)
        } else {
            self.embed_prefixed(&format!("{}{text}", self.spec.query_prefix))
        }
    }
}

impl std::fmt::Debug for MlxEmbedder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MlxEmbedder")
            .field("model", &self.spec.id)
            .finish_non_exhaustive()
    }
}

// No explicit Drop: dropping the struct closes stdin, which ends the sidecar's
// read loop and lets it exit on its own; the child is never reaped because it
// dies with the brain process.
