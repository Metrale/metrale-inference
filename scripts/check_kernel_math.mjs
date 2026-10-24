// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-22: Render every kernel-math entry with KaTeX in strict mode; exit 1 on the first
// class of failure, naming every entry that fails. Input: the JSON the kernel_math test writes
// ({families: {id: {about, numerics, math[]}}, formats: {name: {title, description, math[],
// encode}}}). Prose renders its `$...$` segments inline; `math` and `encode` render as display.
//
//   node scripts/check_kernel_math.mjs <katex.mjs of KaTeX 0.18.4> <table.json>
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [katexPath, tablePath] = process.argv.slice(2);
if (!katexPath || !tablePath) {
  console.error('usage: node scripts/check_kernel_math.mjs <katex.mjs> <table.json>');
  process.exit(2);
}
const katex = (await import(pathToFileURL(katexPath).href)).default;
const table = JSON.parse(readFileSync(tablePath, 'utf8'));
const opts = (displayMode) => ({ displayMode, throwOnError: true, strict: 'error', trust: false, output: 'htmlAndMathml' });
const bad = [];
let count = 0;
const render = (where, tex, display) => {
  count += 1;
  try { katex.renderToString(tex, opts(display)); } catch (e) { bad.push(`${where}: ${e.message}`); }
};
const prose = (where, text) => text.split('$').forEach((p, i) => { if (i % 2) render(where, p, false); });
for (const [id, f] of Object.entries(table.families)) {
  prose(`${id}.about`, f.about);
  prose(`${id}.numerics`, f.numerics);
  f.math.forEach((m, i) => render(`${id}.math[${i}]`, m, true));
}
for (const [name, f] of Object.entries(table.formats)) {
  f.math.forEach((m, i) => render(`${name}.math[${i}]`, m, true));
  if (f.encode) render(`${name}.encode`, f.encode, true);
}
for (const b of bad) console.error(b);
console.log(`${bad.length ? 'FAIL' : 'OK'}: KaTeX ${katex.version} strict, ${count} expressions, ${Object.keys(table.families).length} families, ${Object.keys(table.formats).length} formats${bad.length ? `, ${bad.length} failed` : ''}`);
process.exit(bad.length ? 1 : 0);
