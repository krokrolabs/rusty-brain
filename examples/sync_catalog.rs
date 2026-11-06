//! Maintainer tool: re-sync contracts/mcp-tools.json's schema and hash fields
//! from the live tool definitions, then validate the result.

use rusty_brain::mcp::{canonical_json_bytes, sha256_hex, tool_definitions, validate_catalog};
use serde_json::{Value, json};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "contracts/mcp-tools.json".into());
    let mut catalog: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let tools = tool_definitions();
    let entries = catalog["tools"].as_array_mut().expect("tools array");
    for entry in entries.iter_mut() {
        let name = entry["name"].as_str().expect("tool name").to_string();
        let tool = tools
            .iter()
            .find(|t| t.name.as_ref() == name)
            .expect("catalog tool must exist live");
        let input = Value::Object(tool.input_schema.as_ref().clone());
        let output = Value::Object(
            tool.output_schema
                .as_ref()
                .expect("output schema")
                .as_ref()
                .clone(),
        );
        let annotations = serde_json::to_value(tool.annotations.as_ref().expect("annotations"))?;
        let description = tool
            .description
            .as_ref()
            .map(|value| value.as_ref())
            .unwrap_or_default();
        entry["inputSchema"] = input.clone();
        entry["outputSchema"] = output.clone();
        entry["hashes"]["description"] = json!(sha256_hex(description.as_bytes()));
        entry["hashes"]["inputSchema"] = json!(sha256_hex(&canonical_json_bytes(&input)?));
        entry["hashes"]["outputSchema"] = json!(sha256_hex(&canonical_json_bytes(&output)?));
        entry["hashes"]["annotations"] = json!(sha256_hex(&canonical_json_bytes(&annotations)?));
    }
    validate_catalog(&catalog, &tools)?;
    std::fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&catalog)?),
    )?;
    println!("catalog synced and validated: {path}");
    Ok(())
}
