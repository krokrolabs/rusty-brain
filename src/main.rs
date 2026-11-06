use std::path::PathBuf;
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::new(cli.data_dir, cli.project)?;

    // `path` needs no database and no model: always instant, always works.
    if matches!(cli.cmd, Command::Path) {
        println!("home:     {}", cfg.home.display());
        println!("project:  {}", cfg.project);
        println!("brain:    {}", cfg.db_path().display());
        println!("models:   {}", cfg.models_dir().display());
        println!("exports:  {}", cfg.exports_dir().display());
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
        Command::Recall { query, limit } => {
            let hits = brain.recall(&query, limit).await?;
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
        Command::Path => unreachable!(),
    }
    Ok(())
}
