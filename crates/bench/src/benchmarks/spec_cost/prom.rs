// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The Prometheus text exposition format, read back: the sample lines of one
//! `/metrics` page, and lookup of one series by name and exact label set. Only sample lines
//! (`name value`, `name{labels} value`, either with an optional timestamp) are kept; `#`
//! lines and blank lines are skipped.
//!
//! Owner: bench, spec-cost.
//! Invariants:
//! - A sample line that does not parse fails the whole page, quoting the line.
//! - A lookup that matches more than one series is an error, never a pick.

use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};

/// 2026-10-04: One sample line.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Series {
    pub(crate) name: String,
    pub(crate) labels: BTreeMap<String, String>,
    pub(crate) value: f64,
}

/// 2026-10-04: Every sample line of one page, in page order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Scrape {
    pub(crate) series: Vec<Series>,
}

impl Scrape {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        let mut series = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            series.push(
                parse_line(line).with_context(|| format!("/metrics line {}: {line:?}", i + 1))?,
            );
        }
        Ok(Self { series })
    }

    /// 2026-10-04: The value of the series called `name` whose label set is exactly
    /// `labels` (order does not matter). `None` when no series matches.
    pub(crate) fn value(&self, name: &str, labels: &[(&str, &str)]) -> Result<Option<f64>> {
        let want: BTreeMap<&str, &str> = labels.iter().copied().collect();
        let mut hits = self.series.iter().filter(|s| {
            s.name == name
                && s.labels.len() == want.len()
                && s.labels
                    .iter()
                    .all(|(k, v)| want.get(k.as_str()) == Some(&v.as_str()))
        });
        let first = hits.next().map(|s| s.value);
        if hits.next().is_some() {
            bail!("/metrics carries {name}{labels:?} more than once");
        }
        Ok(first)
    }
}

fn parse_line(line: &str) -> Result<Series> {
    let name_end = line
        .find(|c: char| c == '{' || c.is_whitespace())
        .ok_or_else(|| anyhow!("no value after the metric name"))?;
    let name = &line[..name_end];
    if name.is_empty() {
        bail!("empty metric name");
    }
    let (labels, rest) = if line[name_end..].starts_with('{') {
        parse_labels(&line[name_end + 1..])?
    } else {
        (BTreeMap::new(), &line[name_end..])
    };
    let mut fields = rest.split_whitespace();
    let value_text = fields.next().ok_or_else(|| anyhow!("no value"))?;
    let value: f64 = value_text
        .parse()
        .map_err(|_| anyhow!("value {value_text:?} is not a number"))?;
    // 2026-10-04: The format allows one integer timestamp after the value.
    if let Some(ts) = fields.next() {
        ts.parse::<i64>()
            .map_err(|_| anyhow!("timestamp {ts:?} is not an integer"))?;
    }
    if fields.next().is_some() {
        bail!("text after the value and timestamp");
    }
    Ok(Series {
        name: name.to_string(),
        labels,
        value,
    })
}

/// 2026-10-04: Parse `k="v",k2="v2"}` (the text after `{`) and return the labels and the
/// text after the closing `}`. Values may hold `\\`, `\"` and `\n` escapes, and a `}` or `,`
/// inside quotes is part of the value.
fn parse_labels(text: &str) -> Result<(BTreeMap<String, String>, &str)> {
    let mut labels = BTreeMap::new();
    let mut chars = text.char_indices().peekable();
    loop {
        while chars
            .next_if(|(_, c)| c.is_whitespace() || *c == ',')
            .is_some()
        {}
        let (start, c) = chars
            .next()
            .ok_or_else(|| anyhow!("label set has no closing brace"))?;
        if c == '}' {
            return Ok((labels, &text[start + 1..]));
        }
        let mut key = String::from(c);
        loop {
            match chars.next() {
                Some((_, '=')) => break,
                Some((_, ch)) => key.push(ch),
                None => bail!("label {key:?} has no value"),
            }
        }
        let key = key.trim().to_string();
        if chars.next().map(|(_, ch)| ch) != Some('"') {
            bail!("label {key:?}: value is not quoted");
        }
        let mut value = String::new();
        loop {
            match chars.next() {
                Some((_, '"')) => break,
                Some((_, '\\')) => match chars.next() {
                    Some((_, 'n')) => value.push('\n'),
                    Some((_, ch)) => value.push(ch),
                    None => bail!("label {key:?}: value ends inside an escape"),
                },
                Some((_, ch)) => value.push(ch),
                None => bail!("label {key:?}: value is not terminated"),
            }
        }
        if labels.insert(key.clone(), value).is_some() {
            bail!("label {key:?} appears twice");
        }
    }
}

#[cfg(test)]
#[path = "prom_tests.rs"]
mod tests;
