//! Session identity when the harness does not pass one explicitly.
//!
//! Claude Code gives every MCP server it starts `CLAUDE_CODE_SESSION_ID`, and
//! keeps the name the user gave the session (`/rename`, `claude -n`) as
//! `custom-title` records in that session's local transcript. Reading it lets a
//! memory say which named session wrote it, with no per-session configuration.
//! Everything here is a local file read; nothing is written and nothing leaves
//! the machine. Any failure falls back to the session id, then to no session.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::store::MAX_PROVENANCE_BYTES;

const SESSION_ID_ENV: &str = "CLAUDE_CODE_SESSION_ID";
const TITLE_MARKER: &str = "\"type\":\"custom-title\"";
/// The title is re-recorded often, so the end of the transcript normally has it.
const TAIL_BYTES: u64 = 512 * 1024;

/// The session to record for a write: the explicit value when the harness set
/// one, else the Claude Code session's current name, else its id, else empty.
pub fn resolve(explicit: Option<&str>) -> String {
    if let Some(value) = explicit {
        return value.to_string();
    }
    let Some(id) = std::env::var(SESSION_ID_ENV).ok().filter(|id| is_uuid(id)) else {
        return String::new();
    };
    let title = claude_config_dir().and_then(|dir| claude_code_title(&dir, &id));
    truncate(title.unwrap_or(id))
}

/// The newest name of Claude Code session `id` under config directory `dir`.
pub fn claude_code_title(dir: &Path, id: &str) -> Option<String> {
    if !is_uuid(id) {
        return None;
    }
    let transcript = std::fs::read_dir(dir.join("projects"))
        .ok()?
        .flatten()
        .map(|project| project.path().join(format!("{id}.jsonl")))
        .find(|path| path.is_file())?;
    let mut file = std::fs::File::open(transcript).ok()?;
    let len = file.metadata().ok()?.len();
    if len > TAIL_BYTES {
        file.seek(SeekFrom::Start(len - TAIL_BYTES)).ok()?;
        if let Some(title) = last_title(&mut file) {
            return Some(title);
        }
        file.seek(SeekFrom::Start(0)).ok()?;
    }
    last_title(&mut file)
}

fn last_title(file: &mut std::fs::File) -> Option<String> {
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).ok()?;
    String::from_utf8_lossy(&bytes)
        .lines()
        .rev()
        .filter(|line| line.contains(TITLE_MARKER))
        .find_map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).ok()?;
            let title = record.get("customTitle")?.as_str()?.trim();
            (!title.is_empty()).then(|| title.to_string())
        })
}

fn claude_config_dir() -> Option<PathBuf> {
    match std::env::var_os("CLAUDE_CONFIG_DIR") {
        Some(dir) => Some(PathBuf::from(dir)),
        None => Some(directories::BaseDirs::new()?.home_dir().join(".claude")),
    }
}

/// Only a UUID is ever joined into a path, so the id cannot escape `projects/`.
fn is_uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value).is_ok()
}

fn truncate(mut value: String) -> String {
    let mut end = value.len().min(MAX_PROVENANCE_BYTES);
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value.truncate(end);
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "7037e61f-dc24-47da-98cd-8cd3188ccac5";

    fn transcript(lines: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("projects").join("-some-project");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join(format!("{ID}.jsonl")), lines.join("\n")).unwrap();
        dir
    }

    #[test]
    fn newest_title_wins_after_a_rename() {
        let dir = transcript(&[
            r#"{"type":"custom-title","customTitle":"duckdb-expert","sessionId":"x"}"#,
            r#"{"type":"user","message":"hello"}"#,
            r#"{"type":"custom-title","customTitle":"oss-expert","sessionId":"x"}"#,
            r#"{"type":"assistant","message":"bye"}"#,
        ]);
        assert_eq!(
            claude_code_title(dir.path(), ID).as_deref(),
            Some("oss-expert")
        );
    }

    #[test]
    fn title_before_the_tail_window_is_still_found() {
        let filler = format!(r#"{{"type":"user","message":"{}"}}"#, "x".repeat(4096));
        let mut lines = vec![r#"{"type":"custom-title","customTitle":"early","sessionId":"x"}"#];
        lines.extend(std::iter::repeat_n(filler.as_str(), 200));
        let dir = transcript(&lines);
        assert_eq!(claude_code_title(dir.path(), ID).as_deref(), Some("early"));
    }

    #[test]
    fn unnamed_missing_or_malformed_gives_none() {
        let dir = transcript(&[
            r#"{"type":"user","message":"mentions \"type\":\"custom-title\" in text"}"#,
            r#"{"type":"custom-title","customTitle":""#,
        ]);
        assert_eq!(claude_code_title(dir.path(), ID), None);
        assert_eq!(
            claude_code_title(dir.path(), "00000000-0000-0000-0000-000000000000"),
            None
        );
        assert_eq!(claude_code_title(dir.path(), "../../etc/passwd"), None);
    }

    #[test]
    fn explicit_value_is_kept_even_when_empty() {
        assert_eq!(resolve(Some("dsh-headless")), "dsh-headless");
        assert_eq!(resolve(Some("")), "");
    }

    #[test]
    fn long_titles_are_cut_on_a_character_boundary() {
        let cut = truncate("é".repeat(MAX_PROVENANCE_BYTES));
        assert!(cut.len() <= MAX_PROVENANCE_BYTES);
        assert!(cut.chars().all(|c| c == 'é'));
    }
}
