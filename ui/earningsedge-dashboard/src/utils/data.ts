import Papa from "papaparse";

export async function loadCSV<T = any>(url: string): Promise<T[]> {
  const res = await fetch(url, { cache: "no-store" });

  // If it's truly missing, stop here with a clear error
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url} (HTTP ${res.status})`);
  }

  const text = await res.text();

  // Vite/SPA fallback symptom: missing artifact returns index.html
  const head = text.slice(0, 200).toLowerCase();
  if (head.includes("<!doctype html") || head.includes("<html")) {
    throw new Error(
      `Artifact not found or mis-routed: ${url} returned HTML (did you copy it into public/artifacts?)`
    );
  }

  const parsed = Papa.parse<T>(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: "greedy",
  });

  if (parsed.errors?.length) {
    console.warn("CSV parse warnings:", parsed.errors);
  }

  // Filter out empty rows
  const rows = (parsed.data ?? []).filter((r: any) =>
    r && Object.values(r).some((v) => v !== null && v !== undefined && String(v).trim() !== "")
  );

  return rows as T[];
}
