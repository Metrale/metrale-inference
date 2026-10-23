// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `kg_query`'s read-only guard (design §2.2): the text is parsed by LatticeDB's own
//! Cypher parser and only a pure `Query` statement passes; `CREATE`, `DELETE`, `SET`, and a
//! mutation after a `RETURN` parse to other statement kinds and are refused at the AST, never by
//! regex. Keywords the grammar lacks (`MERGE`, `CALL`, `DROP`, `REMOVE`) fail to parse.
//!
//! Owner: metrale-kg-mcp.
//! Invariants: a refused text never reaches the executor.

use lattice_core::CypherValue;
use lattice_core::cypher::ast::Statement;
use lattice_core::cypher::parser::CypherParser;
use metrale_error::{ErrorCode, Result, error};
use serde_json::Value;
use std::collections::HashMap;

/// 2026-10-23: Accept `cypher` only if it parses to a read-only query.
pub fn read_only(cypher: &str) -> Result<()> {
    let stmt = CypherParser::new()
        .parse(cypher)
        .map_err(|e| error!(ErrorCode::KgCypherRejected, e.to_string()))?;
    match stmt {
        Statement::Query(_) => Ok(()),
        Statement::Create(_) | Statement::Delete(_) | Statement::ReadWrite { .. } => Err(error!(
            ErrorCode::KgCypherRejected,
            "the statement mutates the graph; kg_query is read-only"
        )),
    }
}

/// 2026-10-23: Tool `params` (strings, numbers, booleans) as Cypher values.
pub fn params(v: Option<&Value>) -> Result<HashMap<String, CypherValue>> {
    let mut out = HashMap::new();
    let Some(v) = v else { return Ok(out) };
    let obj = v
        .as_object()
        .ok_or_else(|| error!(ErrorCode::KgToolArgument, "params", "must be an object"))?;
    for (k, x) in obj {
        let c = match x {
            Value::String(s) => CypherValue::from(s.as_str()),
            Value::Bool(b) => CypherValue::Bool(*b),
            Value::Number(n) if n.is_i64() => CypherValue::Int(n.as_i64().unwrap_or(0)),
            Value::Number(n) => CypherValue::Float(n.as_f64().unwrap_or(0.0)),
            _ => {
                return Err(error!(
                    ErrorCode::KgToolArgument,
                    k.clone(),
                    "must be a string, number or boolean"
                ));
            }
        };
        out.insert(k.clone(), c);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutations_are_refused_at_the_ast_and_reads_pass() {
        for q in [
            "CREATE (n:LedgerEntry {id: 'x'})",
            "MATCH (n) DELETE n",
            "MATCH (n) DETACH DELETE n",
            "MATCH (n) SET n.verdict = 'WIN' RETURN n",
            "MATCH (n:LedgerEntry) RETURN n.id CREATE (m:X)",
            "MERGE (n:X)",
            "CALL db.labels()",
            "DROP INDEX x",
        ] {
            assert_eq!(
                read_only(q).unwrap_err().code(),
                ErrorCode::KgCypherRejected,
                "{q}"
            );
        }
        read_only(
            "MATCH (l:LedgerEntry {verdict_kind:'BLOCKED'})-[:BLOCKED_BY]->(g) RETURN l.id, g.id",
        )
        .unwrap();
        read_only("MATCH (l:LedgerEntry)-[:DEPENDS_ON*1..100]->(d) RETURN d.id").unwrap();
    }
}
