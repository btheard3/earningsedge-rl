import { useEffect, useMemo, useState } from "react";
import { loadCSV } from "../utils/data";
import KpiCard from "../components/KpiCard";

type Row = {
  symbol: string;
  n_pairs: number;
  fail_rate: number; // 0..1
  failures?: number; // <-- prefer integer if present
  reason?: string;
  failure_flags?: string;

  mean_delta_eq_vs_buyhold: number;
  mean_dd_improve_vs_buyhold: number;
  mean_delta_eq_vs_avoid: number;
  mean_dd_improve_vs_avoid: number;
};

const CSV_PATH = "/artifacts/sprint4/symbol_failure_summary.csv";

function parseFlags(s?: string) {
  if (!s) return [];
  // handle pipe format first
  if (s.includes("|")) return s.split("|").map(x => x.trim()).filter(Boolean);

  // fallback: python list string like "['a', 'b']"
  return s
    .replace(/^\s*\[/, "")
    .replace(/\]\s*$/, "")
    .split(",")
    .map(x => x.replace(/['"]/g, "").trim())
    .filter(Boolean);
}

function toNum(x: unknown) {
  const n = typeof x === "number" ? x : Number(x);
  return Number.isFinite(n) ? n : NaN;
}

function pct(x: number, digits = 2) {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

function fmt(x: number, digits = 4) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

type SortKey = "failures_desc" | "fail_rate_desc" | "symbol_asc";

export default function ErrorAnalysis() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortKey>("failures_desc");

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        setLoading(true);
        setErr(null);

        const raw = await loadCSV<any>(CSV_PATH);

        const cleaned: Row[] = (raw ?? [])
          .map((r: any) => ({
            symbol: String(r.symbol ?? "").trim(),
            n_pairs: toNum(r.n_pairs),
            fail_rate: toNum(r.fail_rate),
            failures: r.failures !== undefined ? toNum(r.failures) : undefined,
            reason: r.reason ? String(r.reason) : "",
            failure_flags: r.failure_flags ? String(r.failure_flags) : "",

            mean_delta_eq_vs_buyhold: toNum(r.mean_delta_eq_vs_buyhold),
            mean_dd_improve_vs_buyhold: toNum(r.mean_dd_improve_vs_buyhold),
            mean_delta_eq_vs_avoid: toNum(r.mean_delta_eq_vs_avoid),
            mean_dd_improve_vs_avoid: toNum(r.mean_dd_improve_vs_avoid),
          }))
          .filter((r) => r.symbol);

        if (alive) setRows(cleaned);
      } catch (e: any) {
        if (alive) setErr(e?.message ?? "Failed to load error analysis CSV");
      } finally {
        if (alive) setLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  const enriched = useMemo(() => {
    return rows.map((r) => {
      const total = Number.isFinite(r.n_pairs) ? r.n_pairs : NaN;
      const fr = Number.isFinite(r.fail_rate) ? r.fail_rate : NaN;

      // Prefer integer failures if provided by notebook.
      const failures =
        r.failures !== undefined && Number.isFinite(r.failures)
          ? Math.round(r.failures)
          : Number.isFinite(total) && Number.isFinite(fr)
            ? Math.round(fr * total)
            : NaN;

      return { ...r, total, failures, flagsArr: parseFlags(r.failure_flags) };

    });
  }, [rows]);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    const base = q
      ? enriched.filter((r) => r.symbol.toUpperCase().includes(q))
      : enriched;

    const sorted = [...base].sort((a, b) => {
      if (sort === "symbol_asc") return a.symbol.localeCompare(b.symbol);

      if (sort === "fail_rate_desc") {
        const av = Number.isFinite(a.fail_rate) ? a.fail_rate : -Infinity;
        const bv = Number.isFinite(b.fail_rate) ? b.fail_rate : -Infinity;
        return bv - av;
      }

      // failures_desc
      const af = Number.isFinite(a.failures) ? a.failures : -Infinity;
      const bf = Number.isFinite(b.failures) ? b.failures : -Infinity;
      return bf - af;
    });

    return sorted;
  }, [enriched, query, sort]);

  const kpis = useMemo(() => {
    if (!enriched.length) return null;

    const totalFailures = enriched.reduce((acc, r) => {
      const f = Number.isFinite(r.failures) ? r.failures : 0;
      return acc + f;
    }, 0);

    const totalPairs = enriched.reduce((acc, r) => {
      const t = Number.isFinite(r.total) ? r.total : 0;
      return acc + t;
    }, 0);

    const avgFailRate = totalPairs > 0 ? totalFailures / totalPairs : NaN;

    const worstByFailures = [...enriched].sort((a, b) => {
      const af = Number.isFinite(a.failures) ? a.failures : -Infinity;
      const bf = Number.isFinite(b.failures) ? b.failures : -Infinity;
      return bf - af;
    })[0];

    const worstByRate = [...enriched].sort((a, b) => {
      const av = Number.isFinite(a.fail_rate) ? a.fail_rate : -Infinity;
      const bv = Number.isFinite(b.fail_rate) ? b.fail_rate : -Infinity;
      return bv - av;
    })[0];

    return { totalFailures, avgFailRate, worstByFailures, worstByRate };
  }, [enriched]);

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-5">
        <div className="text-sm text-slate-300">Error Analysis</div>
        <div className="text-xs text-slate-400 mt-1">
          Loaded from <span className="font-mono">{CSV_PATH}</span>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-5">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <h2 className="text-sm font-semibold text-slate-200">
            Where episodes fail most
          </h2>

          <div className="flex items-center gap-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by symbol..."
              className="h-9 w-48 rounded-xl border border-slate-800 bg-slate-950/40 px-3 text-sm text-slate-200 placeholder:text-slate-500 outline-none focus:border-slate-600"
            />

            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              className="h-9 rounded-xl border border-slate-800 bg-slate-950/40 px-3 text-sm text-slate-200 outline-none focus:border-slate-600"
            >
              <option value="failures_desc">Sort: Failures (desc)</option>
              <option value="fail_rate_desc">Sort: Failure rate (desc)</option>
              <option value="symbol_asc">Sort: Symbol (A→Z)</option>
            </select>

            <a
              className="text-xs text-slate-400 hover:text-slate-200 underline"
              href={CSV_PATH}
            >
              download CSV
            </a>
          </div>
        </div>

        {loading && <div className="mt-4 text-sm text-slate-400">Loading…</div>}
        {err && (
          <div className="mt-4 text-sm text-red-300">Could not load: {err}</div>
        )}

        {!loading && !err && (
          <>
            {kpis && (
              <div className="mt-5 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <KpiCard
                  title="Total failures"
                  value={kpis.totalFailures}
                  formatter={(v) => (Number.isFinite(v) ? String(Math.round(v)) : "—")}
                />
                <KpiCard
                  title="Avg failure rate"
                  value={kpis.avgFailRate}
                  formatter={(v) => pct(Number(v), 2)}
                />
                <KpiCard
                  title="Worst by failures"
                  policy={kpis.worstByFailures?.symbol ?? "—"}
                  value={Number.isFinite(kpis.worstByFailures?.failures) ? (kpis.worstByFailures!.failures as number) : NaN}
                  formatter={(v) =>
                    Number.isFinite(Number(v)) ? `${Math.round(Number(v))} failures` : "—"
                  }   
                />
                <KpiCard
                  title="Worst by rate"
                  policy={kpis.worstByRate?.symbol ?? "—"}
                  value={kpis.worstByRate?.fail_rate ?? NaN}
                  formatter={(v) => pct(Number(v), 2)}
                />
              </div>
            )}

            <div className="mt-5 overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-sm">
                <thead className="bg-slate-900/60 text-slate-300">
                  <tr>
                    <th className="px-3 py-2 text-left">symbol</th>
                    <th className="px-3 py-2 text-right">failures</th>
                    <th className="px-3 py-2 text-right">n_pairs</th>
                    <th className="px-3 py-2 text-right">fail_rate</th>
                    <th className="px-3 py-2 text-right">Δeq vs buy&hold</th>
                    <th className="px-3 py-2 text-right">DD improve vs buy&hold</th>
                    <th className="px-3 py-2 text-right">Δeq vs avoid</th>
                    <th className="px-3 py-2 text-right">DD improve vs avoid</th>
                    <th className="px-3 py-2 text-left">reason</th>
                    <th className="px-3 py-2 text-left">flags</th>
                  </tr>
                </thead>

                <tbody>
                  {filtered.map((r) => (
                    <tr key={r.symbol} className="border-t border-slate-800">
                      <td className="px-3 py-2 text-slate-200">{r.symbol}</td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {Number.isFinite(r.failures) ? r.failures : "—"}
                      </td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {Number.isFinite(r.n_pairs) ? r.n_pairs : "—"}
                      </td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {pct(r.fail_rate)}
                      </td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {fmt(r.mean_delta_eq_vs_buyhold)}
                      </td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {fmt(r.mean_dd_improve_vs_buyhold)}
                      </td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {fmt(r.mean_delta_eq_vs_avoid)}
                      </td>

                      <td className="px-3 py-2 text-right text-slate-300">
                        {fmt(r.mean_dd_improve_vs_avoid)}
                      </td>

                      <td className="px-3 py-2 text-slate-400">
                        {r.reason && r.reason.trim().length ? r.reason : "—"}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-1">
                        {(r.flagsArr?.length ? r.flagsArr : []).map((f) => (
                        <span key={f} className="rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-0.5 text-xs text-slate-300">{f}
                        </span>
                 ))}
                        {!r.flagsArr?.length && <span className="text-slate-500">—</span>}
                        </div>
                      </td>
                    </tr>
                  ))}

                  {!filtered.length && (
                    <tr>
                      <td
                        className="px-3 py-8 text-center text-slate-500"
                        colSpan={10}
                      >
                        No rows match your filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
