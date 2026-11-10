use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};

use rusty_brain::app::Brain;
use rusty_brain::config::Config;
use rusty_brain::mcp;

/// Local-first memory for AI coding agents. One SQLite (libsql) file per
/// project, embeddings computed locally, no account and no provider keys.
#[derive(Parser)]
#[command(name = "rusty-brain", version, about)]
struct Cli {
    /// Data directory (default ~/.rusty-brain, or $RUSTY_BRAIN_HOME).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    /// Project name (default: derived from the current directory).
    #[arg(long, global = true)]
    project: Option<String>,
    /// Agent identity recorded on memories (default: $RUSTY_BRAIN_AGENT or `cli`).
    #[arg(long, global = true)]
    agent: Option<String>,
    /// Session id/name recorded on memories (default: $RUSTY_BRAIN_SESSION or empty).
    #[arg(long, global = true)]
    session: Option<String>,
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run as an MCP server on stdin/stdout (this is how agent harnesses mount it).
    Serve,
    /// Store a memory.
    Remember {
        content: String,
        #[arg(long)]
        tags: Option<String>,
    },
    /// Hybrid semantic + keyword recall.
    Recall {
        query: String,
        #[arg(short = 'n', long, default_value_t = 5)]
        limit: usize,
        /// Recall scope: `all` (default), `self`, `others`, or an agent name.
        #[arg(long)]
        agent: Option<String>,
    },
    /// Soft-delete a memory by id.
    Forget { id: String },
    /// Most recent memories.
    Recent {
        #[arg(short = 'n', long, default_value_t = 10)]
        limit: usize,
    },
    /// Memory count, model and file size.
    Stats,
    /// Print every path rusty-brain uses.
    Path,
    /// Export memories to a new Parquet file (derived; the brain file stays the truth).
    Export,
    /// Re-embed every memory with another model (preserves ids, tags, provenance).
    Reembed {
        /// Target model id (see `rusty-brain models`).
        model: String,
        /// Delete the previous model's vectors after a successful migration.
        #[arg(long, conflicts_with = "prune_all")]
        prune: bool,
        /// Delete every older model's vectors after a successful migration.
        #[arg(long)]
        prune_all: bool,
    },
    /// Switch to a model whose vectors already cover every active memory.
    UseModel {
        /// Existing model id to select without re-embedding.
        model: String,
    },
    /// List the supported embedding models.
    Models,
    /// Install the optional MLX (Metal) backend: sidecar script, pinned venv,
    /// and the model artifact (idempotent; needs network and uv once).
    SetupMlx,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {}", safe_diagnostic(&error));
            ExitCode::FAILURE
        }
    }
}

fn safe_diagnostic(error: &anyhow::Error) -> String {
    // anyhow's default main termination prints debug chains and, when enabled,
    // stack backtraces. Print only the top-level actionable message instead.
    let mut message = error.to_string();
    for name in [
        "HF_TOKEN",
        "RB_TEST_SECRET",
        "DEEPSEEK_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ] {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            message = message.replace(&value, "[redacted]");
        }
    }
    message
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::new(cli.data_dir, cli.project, cli.agent, cli.session)?;

    // `path` needs no database and no model: always instant, always works.
    if matches!(cli.cmd, Command::Path) {
        println!("home:     {}", cfg.home.display());
        println!("project:  {}", cfg.project);
        println!("brain:    {}", cfg.db_path().display());
        println!("models:   {}", cfg.models_dir().display());
        println!("exports:  {}", cfg.exports_dir().display());
        return Ok(());
    }

    // `setup-mlx` needs no database either; it populates the home directory.
    if matches!(cli.cmd, Command::SetupMlx) {
        rusty_brain::mlx::setup(&cfg.home)?;
        return Ok(());
    }

    let brain = Arc::new(Brain::open(cfg).await?);
    match cli.cmd {
        Command::Serve => mcp::serve(brain).await?,
        Command::Remember { content, tags } => {
            let id = brain
                .remember(&content, tags.as_deref().unwrap_or(""))
                .await?;
            println!("{id}");
        }
        Command::Recall {
            query,
            limit,
            agent,
        } => {
            let hits = brain
                .recall(&query, limit, brain.agent_filter(agent.as_deref()))
                .await?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::Forget { id } => {
            if brain.forget(&id).await? {
                println!("forgotten {id}");
            } else {
                println!("no such memory: {id}");
            }
        }
        Command::Recent { limit } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&brain.recent(limit).await?)?
            );
        }
        Command::Stats => {
            println!("{}", serde_json::to_string_pretty(&brain.stats().await?)?);
        }
        Command::Export => {
            let (path, count) = brain.export_parquet().await?;
            println!("exported {count} memories to {}", path.display());
        }
        Command::Reembed {
            model,
            prune,
            prune_all,
        } => {
            let mode = if prune_all {
                rusty_brain::app::Prune::All
            } else if prune {
                rusty_brain::app::Prune::Old
            } else {
                rusty_brain::app::Prune::Keep
            };
            let report = brain.reembed(&model, mode).await?;
            let note = if report.pruned_models.is_empty() {
                "old vectors kept (use-model switches back without re-embedding)".to_string()
            } else {
                format!("pruned: {}", report.pruned_models.join(", "))
            };
            println!(
                "re-embedded {} memories: {} -> {} ({note})",
                report.memories, report.from_model, report.to_model
            );
        }
        Command::UseModel { model } => {
            let from = brain.use_model(&model).await?;
            if from == model {
                println!("brain already uses {model}");
            } else {
                println!("switched model: {from} -> {model} (existing vectors; no re-embedding)");
            }
        }
        Command::Models => {
            for m in rusty_brain::config::MODELS {
                match m.backend {
                    rusty_brain::config::Backend::Fastembed => {
                        println!("{} (dim {})", m.id, m.dim);
                    }
                    rusty_brain::config::Backend::Mlx => {
                        println!("{} (dim {}, MLX; needs setup-mlx)", m.id, m.dim);
                    }
                }
            }
        }
        Command::Path => unreachable!(),
        Command::SetupMlx => unreachable!("handled before the brain is opened"),
    }
    Ok(())
}
