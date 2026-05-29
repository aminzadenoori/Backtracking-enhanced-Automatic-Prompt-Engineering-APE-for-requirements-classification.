import { useState, useRef, useCallback, useEffect } from "react";

const SAMPLE_CSV = `text,label
The system shall process all login requests within 2 seconds under normal load,Performance
All user passwords must be stored using bcrypt hashing with a salt factor of 12,Security
The dashboard shall display a pie chart summarising ticket status,Functional
The platform shall support 10000 concurrent users without degradation,Scalability
Data transmissions between client and server must use TLS 1.3 or higher,Security
The report export function shall generate a PDF in under 5 seconds,Performance
The API shall expose a REST endpoint for creating new user accounts,Functional
The system shall maintain 99.9 percent uptime measured monthly,Availability
Sensitive fields must be encrypted at rest using AES-256,Security
The search feature shall return results within 1.5 seconds for up to 1 million records,Performance
The application shall allow administrators to configure role-based access,Functional
The service shall auto-scale horizontally when CPU exceeds 80 percent,Scalability
All audit logs must be retained for a minimum of 7 years,Security
The system shall be accessible 24 hours a day 7 days a week,Availability
Users shall be able to export data in CSV and JSON formats,Functional
Database write throughput shall handle 5000 transactions per second,Scalability`;

const MODELS = [
  { id: "claude-sonnet-4-20250514", label: "Claude Sonnet 4" },
  { id: "claude-haiku-4-5-20251001", label: "Claude Haiku 4.5" },
  { id: "claude-opus-4-20250514", label: "Claude Opus 4" },
];

const STRATEGIES = [
  { id: "zero_shot",    label: "Zero-shot",       desc: "Task description + label definitions only" },
  { id: "few_shot",     label: "Few-shot",         desc: "Examples added alongside task description" },
  { id: "cot",          label: "Chain-of-Thought", desc: "Step-by-step reasoning before final label" },
  { id: "cot_few_shot", label: "CoT + Few-shot",   desc: "Examples with reasoning traces" },
];

function parseCSV(raw) {
  const lines = raw.trim().split("\n").filter(Boolean);
  if (lines.length < 2) return null;
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const m = lines[i].match(/^(".*?"|[^,]+),(".*?"|.+)$/);
    if (!m) continue;
    const text  = m[1].replace(/^"|"$/g, "").trim();
    const label = m[2].replace(/^"|"$/g, "").trim();
    if (text && label) rows.push({ text, label });
  }
  return rows.length >= 2 ? rows : null;
}

function computeMetrics(preds, actuals, labels) {
  const perClass = {};
  for (const lbl of labels) {
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < preds.length; i++) {
      const p = (preds[i] || "").toLowerCase().trim();
      const a = (actuals[i] || "").toLowerCase().trim();
      const l = lbl.toLowerCase();
      if (p === l && a === l) tp++;
      else if (p === l && a !== l) fp++;
      else if (p !== l && a === l) fn++;
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec  = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
    perClass[lbl] = { f1: +f1.toFixed(3), precision: +prec.toFixed(3), recall: +rec.toFixed(3), support: actuals.filter(a => a === lbl).length };
  }
  const macro = +(Object.values(perClass).reduce((s, v) => s + v.f1, 0) / labels.length).toFixed(3);
  return { perClass, macroF1: macro };
}

function extractLabel(text, labels) {
  const t = (text || "").toLowerCase().trim();
  for (const line of t.split("\n").reverse()) {
    const s = line.trim();
    if (s.startsWith("label:")) {
      const c = s.slice(6).trim();
      for (const l of labels) if (c === l.toLowerCase()) return l;
    }
  }
  for (const l of labels) if (t === l.toLowerCase()) return l;
  for (const l of labels) if (t.includes(l.toLowerCase())) return l;
  return labels[0];
}

function buildPrompt(strategy, fixedPart, optPart, labels, examples) {
  const labelList = labels.join(", ");
  const base = fixedPart.trim() || `You are a precise text classifier. Classify text into exactly one of: ${labelList}.`;
  const defs = optPart.trim();
  let p = base;
  if (defs) p += `\n\n${defs}`;

  if (strategy === "zero_shot") {
    p += "\n\nOutput only the exact label name, nothing else.";
  } else if (strategy === "few_shot") {
    const ex = examples.map(e => `Text: ${e.text}\nLabel: ${e.label}`).join("\n\n");
    p += `\n\nExamples:\n${ex}\n\nOutput only the exact label name.`;
  } else if (strategy === "cot") {
    p += "\n\nThink step by step:\n1. Identify the core concern.\n2. Match to the best category.\n3. On the final line write: LABEL: <category>";
  } else if (strategy === "cot_few_shot") {
    const ex = examples.map(e => `Text: ${e.text}\nReasoning: This concerns "${e.label}" because it directly addresses that category.\nLABEL: ${e.label}`).join("\n\n");
    p += `\n\nThink step by step, then write LABEL: <category>\n\nExamples:\n${ex}`;
  }
  return p;
}

async function llmCall(model, systemPrompt, userMsg) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: userMsg }],
    }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.content?.map(b => b.text || "").join("") || "";
}

async function votedPredict(model, systemPrompt, texts, labels, votingRuns) {
  const allRuns = [];
  for (let v = 0; v < votingRuns; v++) {
    const run = [];
    for (const text of texts) {
      try {
        const resp = await llmCall(model, systemPrompt, `Text: ${text}`);
        run.push(extractLabel(resp, labels));
      } catch { run.push(labels[0]); }
    }
    allRuns.push(run);
  }
  return texts.map((_, i) => {
    const freq = {};
    for (const run of allRuns) freq[run[i]] = (freq[run[i]] || 0) + 1;
    return Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
  });
}

function Tab({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      background: "none", border: "none", padding: "8px 16px", fontSize: 13,
      cursor: "pointer", borderBottom: active ? "2px solid var(--color-text-primary)" : "2px solid transparent",
      color: active ? "var(--color-text-primary)" : "var(--color-text-tertiary)",
      fontWeight: active ? 500 : 400, marginBottom: -1,
    }}>{label}</button>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: 10, padding: "14px 16px", ...style,
    }}>{children}</div>
  );
}

function SLabel({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 8, marginTop: 14 }}>{children}</div>;
}

function F1Bar({ label, value, color }) {
  const pct = +(value * 100).toFixed(1);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
      <span style={{ width: 110, fontSize: 12, color: "var(--color-text-secondary)", textAlign: "right", flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={label}>{label}</span>
      <div style={{ flex: 1, height: 18, background: "var(--color-background-secondary)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3, transition: "width 0.7s ease", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 4, minWidth: pct > 0 ? 24 : 0 }}>
          {pct > 10 && <span style={{ fontSize: 10, color: "#fff", fontWeight: 600 }}>{pct}%</span>}
        </div>
      </div>
    </div>
  );
}

function MetricsTable({ metrics, labels }) {
  if (!metrics) return null;
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, tableLayout: "fixed" }}>
        <thead>
          <tr>
            {["Label","Precision","Recall","F1","Support"].map(h => (
              <th key={h} style={{ textAlign: h === "Label" ? "left" : "center", padding: "5px 6px", color: "var(--color-text-tertiary)", fontWeight: 500, borderBottom: "0.5px solid var(--color-border-tertiary)", fontSize: 11 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map(lbl => {
            const m = metrics.perClass[lbl] || {};
            const f1 = m.f1 || 0;
            const col = f1 >= 0.7 ? "var(--color-text-success)" : f1 >= 0.4 ? "var(--color-text-warning)" : "var(--color-text-danger)";
            return (
              <tr key={lbl}>
                <td style={{ padding: "5px 6px", color: "var(--color-text-primary)", borderBottom: "0.5px solid var(--color-border-tertiary)", fontSize: 12 }}>{lbl}</td>
                <td style={{ textAlign: "center", padding: "5px 6px", borderBottom: "0.5px solid var(--color-border-tertiary)", color: "var(--color-text-secondary)" }}>{((m.precision || 0)*100).toFixed(0)}%</td>
                <td style={{ textAlign: "center", padding: "5px 6px", borderBottom: "0.5px solid var(--color-border-tertiary)", color: "var(--color-text-secondary)" }}>{((m.recall || 0)*100).toFixed(0)}%</td>
                <td style={{ textAlign: "center", padding: "5px 6px", borderBottom: "0.5px solid var(--color-border-tertiary)", color: col, fontWeight: 500 }}>{(f1*100).toFixed(1)}%</td>
                <td style={{ textAlign: "center", padding: "5px 6px", borderBottom: "0.5px solid var(--color-border-tertiary)", color: "var(--color-text-tertiary)" }}>{m.support || 0}</td>
              </tr>
            );
          })}
          <tr>
            <td style={{ padding: "5px 6px", fontWeight: 500, color: "var(--color-text-primary)", fontSize: 12 }}>Macro F1</td>
            <td colSpan={3} style={{ textAlign: "center", padding: "5px 6px", fontWeight: 500, color: "var(--color-text-primary)" }}>{((metrics.macroF1 || 0)*100).toFixed(1)}%</td>
            <td />
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default function APETool() {
  const [tab, setTab] = useState(0);
  const [csvText, setCsvText] = useState("");
  const [rows, setRows] = useState(null);
  const [parseErr, setParseErr] = useState("");
  const [poolPct, setPoolPct] = useState(30);
  const [valPct, setValPct] = useState(30);
  const [model, setModel] = useState(MODELS[0].id);
  const [votingRuns, setVotingRuns] = useState(1);
  const [maxIter, setMaxIter] = useState(8);
  const [backtrackThr, setBacktrackThr] = useState(3);
  const [fixedPrompt, setFixedPrompt] = useState("");
  const [optPrompt, setOptPrompt] = useState("");
  const [selectedStrategy, setSelectedStrategy] = useState("zero_shot");
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState([]);
  const [results, setResults] = useState({});
  const [assembledPrompt, setAssembledPrompt] = useState("");
  const [apeHistory, setApeHistory] = useState([]);
  const [bestPrompt, setBestPrompt] = useState("");
  const [curOpt, setCurOpt] = useState("");
  const [apeInited, setApeInited] = useState(false);
  const [apeValMetrics, setApeValMetrics] = useState(null);
  const logRef = useRef(null);
  const stateRef = useRef({ rows: [], labels: [], poolRows: [], valRows: [], testRows: [], examples: [], best_f1: 0, best_prompt: "", cur_prompt: "", cur_opt: "", ranked: [], no_improve: 0, ptr: 0, iter_n: 0 });
  const abortRef = useRef(false);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log]);

  const addLog = useCallback((msg, type = "info") => {
    setLog(prev => [...prev, { msg, type, ts: new Date().toLocaleTimeString() }]);
  }, []);

  const labels = rows ? [...new Set(rows.map(r => r.label))].sort() : [];
  const labelCounts = rows ? Object.fromEntries(labels.map(l => [l, rows.filter(r => r.label === l).length])) : {};
  const COLORS = ["#6366f1","#0ea5e9","#f59e0b","#10b981","#ec4899","#8b5cf6","#f97316"];

  const handleParse = () => {
    setParseErr("");
    const parsed = parseCSV(csvText);
    if (!parsed) { setParseErr("Could not parse. Need: text column first, label column last, one header row."); return; }
    setRows(parsed);
    const lbls = [...new Set(parsed.map(r => r.label))].sort();
    const n = parsed.length;
    const poolN = Math.max(1, Math.floor(n * poolPct / 100));
    const valN  = Math.max(1, Math.floor(n * valPct / 100));
    const poolRows = parsed.slice(0, poolN);
    const valRows  = parsed.slice(poolN, poolN + valN);
    const testRows = parsed.slice(poolN + valN);
    const byLabel = {};
    for (const l of lbls) byLabel[l] = poolRows.filter(r => r.label === l);
    const examples = lbls.flatMap(l => byLabel[l].slice(0, 2));
    stateRef.current = { ...stateRef.current, rows: parsed, labels: lbls, poolRows, valRows, testRows, examples, best_f1: 0, best_prompt: "", cur_prompt: "", cur_opt: "", ranked: [], no_improve: 0, ptr: 0, iter_n: 0 };
  };

  const runStrategy = async (strategy) => {
    const { labels, testRows, examples } = stateRef.current;
    if (!testRows.length) return;
    const prompt = buildPrompt(strategy, fixedPrompt, optPrompt, labels, examples);
    setAssembledPrompt(prompt);
    addLog(`Running ${STRATEGIES.find(s => s.id === strategy)?.label}…`, "phase");
    const preds = await votedPredict(model, prompt, testRows.map(r => r.text), labels, votingRuns);
    const metrics = computeMetrics(preds, testRows.map(r => r.label), labels);
    setResults(prev => ({ ...prev, [strategy]: { prompt, metrics } }));
    addLog(`✓ Macro F1: ${(metrics.macroF1 * 100).toFixed(1)}%`, "success");
    return metrics;
  };

  const handleRunOne = async () => {
    if (!rows) return;
    setRunning(true); abortRef.current = false;
    try { await runStrategy(selectedStrategy); } catch(e) { addLog(`Error: ${e.message}`, "error"); }
    setRunning(false);
  };

  const handleRunAll = async () => {
    if (!rows) return;
    setRunning(true); abortRef.current = false;
    for (const s of STRATEGIES) {
      if (abortRef.current) break;
      try { await runStrategy(s.id); } catch(e) { addLog(`Error: ${e.message}`, "error"); }
    }
    setRunning(false);
    setTab(4);
  };

  const handleApeInit = async () => {
    if (!rows) return;
    setRunning(true); abortRef.current = false;
    setApeHistory([]); setLog([]); setApeValMetrics(null);
    const S = stateRef.current;
    const prompt = buildPrompt("few_shot", fixedPrompt, optPrompt, S.labels, S.examples);
    S.cur_prompt = prompt; S.cur_opt = optPrompt; S.iter_n = 0; S.no_improve = 0; S.ptr = 0; S.ranked = [];
    addLog("APE init — evaluating starting prompt on validation set…", "phase");
    try {
      const preds = await votedPredict(model, prompt, S.valRows.map(r => r.text), S.labels, votingRuns);
      const metrics = computeMetrics(preds, S.valRows.map(r => r.label), S.labels);
      S.best_f1 = metrics.macroF1; S.best_prompt = prompt;
      S.ranked = [{ prompt, f1: metrics.macroF1 }];
      setBestPrompt(prompt); setCurOpt(optPrompt);
      setResults(prev => ({ ...prev }));
      addLog(`Starting Val F1: ${(metrics.macroF1 * 100).toFixed(1)}% — click Next iteration to optimise.`, "success");
      setApeInited(true);
    } catch(e) { addLog(`Init error: ${e.message}`, "error"); }
    setRunning(false);
  };

  const handleApeStep = async () => {
    const S = stateRef.current;
    if (!S.cur_prompt) { addLog("Run Init first.", "error"); return; }
    if (S.iter_n >= maxIter) { addLog(`Max iterations (${maxIter}) reached.`, "info"); return; }
    setRunning(true); abortRef.current = false;
    S.iter_n++;
    const n = S.iter_n;
    addLog(`Iteration ${n}/${maxIter} — best: ${(S.best_f1*100).toFixed(1)}%`, "iter");

    const metaSys = "You are an expert NLP prompt engineer. Return ONLY the improved definitions/instructions text. No explanation, no markdown fences.";
    const metaUser = `Improve this text classification prompt to increase F1 score.\n\nCurrent prompt:\n---\n${S.cur_prompt}\n---\n\nCategories: ${S.labels.join(", ")}\n\nRewrite only this section (definitions/rules):\n---\n${S.cur_opt}\n---\n\nReturn ONLY the rewritten section.`;

    let newOpt = S.cur_opt;
    try {
      const raw = await llmCall(model, metaSys, metaUser);
      newOpt = raw.replace(/^```[\w]*\n?/, "").replace(/\n?```$/, "").trim();
      if (newOpt.length < 20) newOpt = S.cur_opt;
    } catch(e) { addLog(`Meta-prompt error: ${e.message}`, "error"); }

    const newPrompt = buildPrompt("few_shot", fixedPrompt, newOpt, S.labels, S.examples);
    let newF1 = 0, newMetrics = null;
    try {
      const preds = await votedPredict(model, newPrompt, S.valRows.map(r => r.text), S.labels, votingRuns);
      newMetrics = computeMetrics(preds, S.valRows.map(r => r.label), S.labels);
      newF1 = newMetrics.macroF1;
    } catch(e) { addLog(`Eval error: ${e.message}`, "error"); setRunning(false); return; }

    S.ranked.push({ prompt: newPrompt, f1: newF1 });
    S.ranked.sort((a, b) => b.f1 - a.f1);

    const improved = newF1 >= S.best_f1;
    let backtracked = false;

    if (improved) {
      addLog(`↑ Improved: ${(S.best_f1*100).toFixed(1)}% → ${(newF1*100).toFixed(1)}%`, "success");
      S.best_f1 = newF1; S.best_prompt = newPrompt;
      S.cur_prompt = newPrompt; S.cur_opt = newOpt;
      S.no_improve = 0; S.ptr = 0;
      setBestPrompt(newPrompt); setCurOpt(newOpt);
    } else {
      S.no_improve++;
      addLog(`→ ${(newF1*100).toFixed(1)}% (best: ${(S.best_f1*100).toFixed(1)}%)`, "info");
    }

    if (S.no_improve >= backtrackThr) {
      const ni = S.ptr + 1;
      if (ni < S.ranked.length) {
        S.cur_prompt = S.ranked[ni].prompt; S.ptr++; S.no_improve = 0;
        backtracked = true;
        addLog(`↩ Backtrack to rank-${ni+1} (F1 ${(S.ranked[ni].f1*100).toFixed(1)}%)`, "warn");
      }
    }

    setApeHistory(prev => [...prev, { n, f1: newF1, improved, backtracked }]);
    if (newMetrics) setApeValMetrics(newMetrics);
    setRunning(false);
  };

  const finalTestEval = async () => {
    const S = stateRef.current;
    if (!S.best_prompt || !S.testRows.length) return;
    addLog("Final evaluation on held-out test set…", "phase");
    try {
      const preds = await votedPredict(model, S.best_prompt, S.testRows.map(r => r.text), S.labels, votingRuns);
      const metrics = computeMetrics(preds, S.testRows.map(r => r.label), S.labels);
      setResults(prev => ({ ...prev, optimized: { prompt: S.best_prompt, metrics, isTest: true } }));
      addLog(`✓ Final Test F1: ${(metrics.macroF1 * 100).toFixed(1)}%`, "success");
    } catch(e) { addLog(`Final test error: ${e.message}`, "error"); }
  };

  const handleApeAll = async () => {
    await handleApeInit();
    for (let i = 0; i < maxIter; i++) {
      if (abortRef.current) break;
      await handleApeStep();
    }
    if (!abortRef.current) await finalTestEval();
  };

  const allResults = [
    ...STRATEGIES.map(s => results[s.id] ? { id: s.id, label: s.label, color: COLORS[STRATEGIES.indexOf(s)], metrics: results[s.id].metrics } : null),
    results.optimized ? { id: "optimized", label: "APE Optimized", color: "#ec4899", metrics: results.optimized.metrics } : null,
  ].filter(Boolean);

  const TABS = ["1 · Setup", "2 · Prompt", "3 · Baselines", "4 · APE", "5 · Results"];

  return (
    <div style={{ fontFamily: "var(--font-sans)", maxWidth: 820, margin: "0 auto", padding: "1.25rem 1rem" }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 18, fontWeight: 500, margin: "0 0 3px" }}>APE Classification Tool</h2>
        <p style={{ fontSize: 13, color: "var(--color-text-tertiary)", margin: 0 }}>
          Automatic prompt engineering for text classification · powered by Claude
        </p>
      </div>

      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 20 }}>
        {TABS.map((t, i) => <Tab key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {tab === 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <SLabel>Dataset</SLabel>
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 6 }}>
              <button onClick={() => setCsvText(SAMPLE_CSV)} style={{ fontSize: 12, color: "var(--color-text-info)", background: "none", border: "none", cursor: "pointer", padding: 0 }}>
                load sample dataset →
              </button>
            </div>
            <textarea value={csvText} onChange={e => setCsvText(e.target.value)} rows={6}
              placeholder={"text,label\n\"The system shall respond in 2 seconds\",Performance\n\"Passwords must be encrypted\",Security"}
              style={{ width: "100%", boxSizing: "border-box", fontFamily: "var(--font-mono)", fontSize: 12, resize: "vertical" }} />
            <div style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "center", flexWrap: "wrap" }}>
              <button onClick={handleParse} style={{ padding: "6px 14px", borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", cursor: "pointer", fontSize: 13 }}>Parse CSV</button>
              {parseErr && <span style={{ fontSize: 12, color: "var(--color-text-danger)" }}>{parseErr}</span>}
              {rows && <span style={{ fontSize: 12, color: "var(--color-text-success)" }}>✓ {rows.length} samples · {labels.length} classes</span>}
            </div>
            {rows && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
                {labels.map((l, i) => (
                  <span key={l} style={{ padding: "2px 10px", borderRadius: 99, fontSize: 11, fontWeight: 500, background: COLORS[i % COLORS.length] + "18", color: COLORS[i % COLORS.length], border: `0.5px solid ${COLORS[i % COLORS.length]}44` }}>
                    {l} ({labelCounts[l]})
                  </span>
                ))}
              </div>
            )}
          </Card>

          <Card>
            <SLabel>Model</SLabel>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {MODELS.map(m => (
                <button key={m.id} onClick={() => setModel(m.id)} style={{
                  padding: "6px 14px", borderRadius: 8, cursor: "pointer", fontSize: 13,
                  border: model === m.id ? "2px solid #6366f1" : "0.5px solid var(--color-border-secondary)",
                  background: model === m.id ? "#eeedfe" : "var(--color-background-primary)",
                  color: model === m.id ? "#3c3489" : "var(--color-text-primary)", fontWeight: model === m.id ? 500 : 400,
                }}>{m.label}</button>
              ))}
            </div>

            <SLabel>Run settings</SLabel>
            {[
              { label: "Example pool %", val: poolPct, set: v => { setPoolPct(v); if (rows) handleParse(); }, min: 10, max: 80, step: 5 },
              { label: "Validation %", val: valPct, set: v => { setValPct(v); if (rows) handleParse(); }, min: 10, max: 80, step: 5 },
              { label: "Voting runs", val: votingRuns, set: setVotingRuns, min: 1, max: 5, step: 1 },
              { label: "APE max iterations", val: maxIter, set: setMaxIter, min: 1, max: 20, step: 1 },
              { label: "Backtrack threshold X", val: backtrackThr, set: setBacktrackThr, min: 1, max: 10, step: 1 },
            ].map(({ label, val, set, min, max, step }) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: "var(--color-text-secondary)", width: 160, flexShrink: 0 }}>{label}</span>
                <input type="range" min={min} max={max} step={step} value={val} onChange={e => set(+e.target.value)} style={{ flex: 1 }} />
                <span style={{ fontSize: 12, fontWeight: 500, width: 28, textAlign: "right" }}>{val}</span>
              </div>
            ))}
            <div style={{ fontSize: 12, color: poolPct + valPct >= 100 ? "var(--color-text-danger)" : "var(--color-text-tertiary)", marginTop: 4 }}>
              Test set = remainder ({Math.max(0, 100 - poolPct - valPct)}%)
              {poolPct + valPct >= 100 && " — pool + validation must be under 100%"}
            </div>
            {rows && poolPct + valPct < 100 && (() => {
              const n = rows.length;
              const poolN = Math.max(1, Math.floor(n * poolPct / 100));
              const valN = Math.max(1, Math.floor(n * valPct / 100));
              const testN = n - poolN - valN;
              return (
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 4 }}>
                  Pool: <strong>{poolN}</strong> · Validation: <strong>{valN}</strong> · Test: <strong>{testN}</strong> samples
                </div>
              );
            })()}
          </Card>
        </div>
      )}

      {tab === 1 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12, lineHeight: 1.6 }}>
              <strong style={{ fontWeight: 500 }}>Fixed part</strong> — task framing and output format. Stays constant across all runs.<br />
              <strong style={{ fontWeight: 500 }}>Optimisable part</strong> — label definitions and disambiguation rules. APE rewrites only this section. Leave blank for defaults.
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 5 }}>Fixed prompt</div>
                <textarea value={fixedPrompt} onChange={e => setFixedPrompt(e.target.value)} rows={8}
                  placeholder={"You are a precise text classifier.\nClassify each text into exactly one of the provided categories.\nOutput only the exact label name, nothing else."}
                  style={{ width: "100%", boxSizing: "border-box", fontSize: 12, resize: "vertical" }} />
              </div>
              <div>
                <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 5 }}>Optimisable prompt <span style={{ color: "var(--color-text-tertiary)" }}>(APE rewrites this)</span></div>
                <textarea value={optPrompt} onChange={e => setOptPrompt(e.target.value)} rows={8}
                  placeholder={"Performance: speed, latency, throughput, response time.\nSecurity: authentication, encryption, access control.\nFunctional: user-visible features and behaviours.\nScalability: concurrent users, load, auto-scaling.\nAvailability: uptime, reliability, fault tolerance."}
                  style={{ width: "100%", boxSizing: "border-box", fontSize: 12, resize: "vertical", borderLeft: "2px solid #6366f1", borderRadius: "0 8px 8px 0" }} />
              </div>
            </div>
          </Card>
        </div>
      )}

      {tab === 2 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
              {STRATEGIES.map((s, i) => (
                <button key={s.id} onClick={() => setSelectedStrategy(s.id)} style={{
                  padding: "6px 12px", borderRadius: 8, cursor: "pointer", fontSize: 12,
                  border: selectedStrategy === s.id ? `2px solid ${COLORS[i]}` : "0.5px solid var(--color-border-secondary)",
                  background: selectedStrategy === s.id ? COLORS[i] + "12" : "var(--color-background-secondary)",
                  color: selectedStrategy === s.id ? COLORS[i] : "var(--color-text-secondary)",
                  fontWeight: selectedStrategy === s.id ? 500 : 400,
                }}>
                  {s.label}
                </button>
              ))}
            </div>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 12 }}>
              {STRATEGIES.find(s => s.id === selectedStrategy)?.desc}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={handleRunOne} disabled={!rows || running} style={{
                padding: "7px 18px", borderRadius: 8, border: "none", cursor: rows && !running ? "pointer" : "default",
                background: rows && !running ? "#6366f1" : "var(--color-background-secondary)",
                color: rows && !running ? "#fff" : "var(--color-text-tertiary)", fontSize: 13, fontWeight: 500,
              }}>Run selected</button>
              <button onClick={handleRunAll} disabled={!rows || running} style={{
                padding: "7px 18px", borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", cursor: rows && !running ? "pointer" : "default",
                background: "var(--color-background-primary)", color: rows && !running ? "var(--color-text-primary)" : "var(--color-text-tertiary)", fontSize: 13,
              }}>Run all 4 baselines</button>
              {running && <button onClick={() => { abortRef.current = true; setRunning(false); }} style={{ padding: "7px 14px", borderRadius: 8, border: "0.5px solid var(--color-border-danger)", background: "none", color: "var(--color-text-danger)", cursor: "pointer", fontSize: 12 }}>Stop</button>}
            </div>
          </Card>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 10 }}>Assembled prompt sent to LLM</div>
              <textarea value={assembledPrompt} readOnly rows={10} style={{ width: "100%", boxSizing: "border-box", fontSize: 11, fontFamily: "var(--font-mono)", resize: "vertical", background: "var(--color-background-secondary)", color: "var(--color-text-primary)" }} />
            </Card>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 10 }}>Results</div>
              {STRATEGIES.map((s, i) => results[s.id] ? (
                <div key={s.id} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 500, color: COLORS[i] }}>{s.label}</span>
                    <span style={{ fontSize: 12, fontWeight: 500, color: COLORS[i] }}>{(results[s.id].metrics.macroF1 * 100).toFixed(1)}%</span>
                  </div>
                  {labels.map(lbl => <F1Bar key={lbl} label={lbl} value={results[s.id].metrics.perClass[lbl]?.f1 || 0} color={COLORS[i]} />)}
                </div>
              ) : null)}
            </Card>
          </div>

          <Card>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", maxHeight: 160, overflowY: "auto" }} ref={logRef}>
              {log.length === 0 ? <span style={{ color: "var(--color-text-tertiary)" }}>Log will appear here…</span> : log.map((e, i) => (
                <div key={i} style={{ marginBottom: 2, color: e.type === "success" ? "var(--color-text-success)" : e.type === "error" ? "var(--color-text-danger)" : e.type === "phase" ? "var(--color-text-info)" : "var(--color-text-secondary)" }}>
                  <span style={{ opacity: 0.4, marginRight: 8 }}>{e.ts}</span>{e.msg}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === 3 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12, lineHeight: 1.6 }}>
              <strong style={{ fontWeight: 500 }}>Interactive:</strong> Init once, then step through one iteration at a time.<br />
              <strong style={{ fontWeight: 500 }}>Automatic:</strong> Run all iterations to completion.
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={handleApeInit} disabled={!rows || running} style={{ padding: "7px 16px", borderRadius: 8, border: "none", background: rows && !running ? "#6366f1" : "var(--color-background-secondary)", color: rows && !running ? "#fff" : "var(--color-text-tertiary)", cursor: rows && !running ? "pointer" : "default", fontSize: 13, fontWeight: 500 }}>① Init APE</button>
              <button onClick={handleApeStep} disabled={!apeInited || running} style={{ padding: "7px 16px", borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: apeInited && !running ? "var(--color-text-primary)" : "var(--color-text-tertiary)", cursor: apeInited && !running ? "pointer" : "default", fontSize: 13 }}>② Next iteration</button>
              <button onClick={handleApeAll} disabled={!rows || running} style={{ padding: "7px 16px", borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: rows && !running ? "var(--color-text-primary)" : "var(--color-text-tertiary)", cursor: rows && !running ? "pointer" : "default", fontSize: 13 }}>Run all automatically</button>
              <button onClick={async () => { setRunning(true); abortRef.current = false; await finalTestEval(); setRunning(false); }} disabled={!apeInited || running} style={{ padding: "7px 16px", borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: apeInited && !running ? "var(--color-text-primary)" : "var(--color-text-tertiary)", cursor: apeInited && !running ? "pointer" : "default", fontSize: 13 }}>Final test eval</button>
              {running && <button onClick={() => { abortRef.current = true; setRunning(false); }} style={{ padding: "7px 14px", borderRadius: 8, border: "0.5px solid var(--color-border-danger)", background: "none", color: "var(--color-text-danger)", cursor: "pointer", fontSize: 12 }}>Stop</button>}
            </div>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 10, lineHeight: 1.5 }}>
              In-loop scoring uses the <strong style={{ fontWeight: 500 }}>validation</strong> set; the held-out <strong style={{ fontWeight: 500 }}>test</strong> set is evaluated once on the final prompt. Each iteration samples balanced examples from the pool: 1 correct-positive, 1 correct-negative, 1 misclassified-positive, 1 misclassified-negative.
            </div>
          </Card>

          {apeHistory.length > 0 && (
            <Card>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 10 }}>Iteration history</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
                {apeHistory.map(h => (
                  <div key={h.n} style={{
                    width: 44, height: 44, borderRadius: 6, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    background: h.improved ? "var(--color-background-success)" : h.backtracked ? "var(--color-background-warning)" : "var(--color-background-secondary)",
                    border: `0.5px solid ${h.improved ? "var(--color-border-success)" : h.backtracked ? "var(--color-border-warning)" : "var(--color-border-tertiary)"}`,
                  }}>
                    <span style={{ fontSize: 9, color: "var(--color-text-tertiary)" }}>#{h.n}</span>
                    <span style={{ fontSize: 12, fontWeight: 500, color: h.improved ? "var(--color-text-success)" : h.backtracked ? "var(--color-text-warning)" : "var(--color-text-secondary)" }}>
                      {(h.f1 * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--color-text-tertiary)" }}>
                <span>✅ improved</span><span>↩ backtracked</span>
              </div>
            </Card>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 8 }}>Best prompt so far</div>
              <textarea value={bestPrompt} readOnly rows={8} style={{ width: "100%", boxSizing: "border-box", fontSize: 11, fontFamily: "var(--font-mono)", resize: "vertical", background: "var(--color-background-secondary)" }} />
            </Card>
            <Card>
              <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>Optimisable section <span style={{ fontWeight: 400, color: "var(--color-text-tertiary)" }}>(evolving)</span></div>
              <textarea value={curOpt} readOnly rows={5} style={{ width: "100%", boxSizing: "border-box", fontSize: 11, fontFamily: "var(--font-mono)", resize: "vertical", background: "var(--color-background-secondary)", borderLeft: "2px solid #6366f1", borderRadius: "0 6px 6px 0", marginBottom: 10 }} />
              {apeValMetrics && (
                <>
                  <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 6 }}>Validation F1 <span style={{ fontWeight: 400, color: "var(--color-text-tertiary)" }}>(in-loop)</span></div>
                  <MetricsTable metrics={apeValMetrics} labels={labels} />
                </>
              )}
              {results.optimized?.isTest && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-success)", marginBottom: 6 }}>Final Test F1 <span style={{ fontWeight: 400, color: "var(--color-text-tertiary)" }}>(held-out)</span></div>
                  <MetricsTable metrics={results.optimized.metrics} labels={labels} />
                </div>
              )}
            </Card>
          </div>

          <Card>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)", maxHeight: 140, overflowY: "auto" }} ref={logRef}>
              {log.length === 0 ? <span style={{ color: "var(--color-text-tertiary)" }}>Log will appear here…</span> : log.map((e, i) => (
                <div key={i} style={{ marginBottom: 2, color: e.type === "success" ? "var(--color-text-success)" : e.type === "error" ? "var(--color-text-danger)" : e.type === "warn" ? "var(--color-text-warning)" : e.type === "iter" ? "var(--color-text-info)" : "var(--color-text-secondary)" }}>
                  <span style={{ opacity: 0.4, marginRight: 8 }}>{e.ts}</span>{e.msg}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === 4 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {allResults.length === 0 ? (
            <div style={{ color: "var(--color-text-tertiary)", fontSize: 14, textAlign: "center", padding: "3rem 0" }}>Run baselines or APE first.</div>
          ) : (
            <>
              <Card>
                <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 14 }}>Macro F1 comparison</div>
                {allResults.map(r => (
                  <F1Bar key={r.id} label={r.label} value={r.metrics.macroF1} color={r.color} />
                ))}
              </Card>

              {labels.map((lbl, li) => (
                <Card key={lbl}>
                  <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10, color: COLORS[li % COLORS.length] }}>{lbl}</div>
                  {allResults.map(r => (
                    <F1Bar key={r.id} label={r.label} value={r.metrics.perClass[lbl]?.f1 || 0} color={r.color} />
                  ))}
                </Card>
              ))}

              <Card>
                <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)", marginBottom: 12 }}>Full results table</div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, tableLayout: "fixed" }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: "left", padding: "5px 8px", color: "var(--color-text-tertiary)", fontWeight: 500, borderBottom: "0.5px solid var(--color-border-tertiary)", width: 130 }}>Method</th>
                        <th style={{ textAlign: "center", padding: "5px 4px", color: "var(--color-text-tertiary)", fontWeight: 500, borderBottom: "0.5px solid var(--color-border-tertiary)" }}>Macro F1</th>
                        {labels.map(l => <th key={l} style={{ textAlign: "center", padding: "5px 4px", color: "var(--color-text-tertiary)", fontWeight: 500, borderBottom: "0.5px solid var(--color-border-tertiary)", fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={l}>{l}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {allResults.map(r => (
                        <tr key={r.id} style={{ background: r.id === "optimized" ? "var(--color-background-secondary)" : "transparent" }}>
                          <td style={{ padding: "5px 8px", color: "var(--color-text-primary)", borderBottom: "0.5px solid var(--color-border-tertiary)", fontWeight: r.id === "optimized" ? 500 : 400, fontSize: 12 }}>{r.label}</td>
                          <td style={{ textAlign: "center", padding: "5px 4px", borderBottom: "0.5px solid var(--color-border-tertiary)", fontWeight: 500, color: r.color }}>{(r.metrics.macroF1 * 100).toFixed(1)}%</td>
                          {labels.map(lbl => {
                            const f1 = r.metrics.perClass[lbl]?.f1 || 0;
                            return <td key={lbl} style={{ textAlign: "center", padding: "5px 4px", borderBottom: "0.5px solid var(--color-border-tertiary)", fontSize: 12, color: f1 >= 0.7 ? "var(--color-text-success)" : f1 >= 0.4 ? "var(--color-text-warning)" : "var(--color-text-danger)" }}>{(f1*100).toFixed(0)}%</td>;
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </div>
      )}
    </div>
  );
}
