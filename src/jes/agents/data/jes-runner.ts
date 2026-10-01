// Calls `uvx jes@__JES_VERSION__ hook` on stdin. uvx ships with uv, so there is no separate jes install.
// `jes runner-settings` prints this file with the version filled in. Print it again after upgrading jes.
import { spawnSync } from "node:child_process";

export type JesDecision = { ok: boolean; onward: string; decision: string };

export type JesRunner = (body: Record<string, unknown>) => JesDecision;

// The agent waits for every check, so a check that hangs blocks after this long.
const TIMEOUT_MS = 60_000;
const UNANSWERED: JesDecision = {
  ok: false,
  onward: "Blocked: jes did not answer.",
  decision: "block",
};

export function spawnJes(body: Record<string, unknown>): JesDecision {
  const run = spawnSync("uvx", ["jes@__JES_VERSION__", "hook"], {
    input: JSON.stringify(body),
    encoding: "utf8",
    timeout: TIMEOUT_MS,
    maxBuffer: 16 * 1024 * 1024,
  });
  return parseDecision(run.stdout);
}

// A jes that crashed, timed out, or is misconfigured prints nothing usable. That blocks.
export function parseDecision(stdout: string | null | undefined): JesDecision {
  const text = (stdout ?? "").trim();
  if (!text) {
    return UNANSWERED;
  }
  try {
    const parsed = JSON.parse(text) as Partial<JesDecision>;
    if (typeof parsed.ok !== "boolean" || typeof parsed.onward !== "string") {
      return UNANSWERED;
    }
    return { ok: parsed.ok, onward: parsed.onward, decision: parsed.ok ? "allow" : "block" };
  } catch {
    return UNANSWERED;
  }
}

export function sessionOf(value: string | undefined, fallback: string): string {
  const cleaned = (value ?? fallback).replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 200);
  return /^[A-Za-z0-9]/.test(cleaned) ? cleaned : `s${cleaned}`;
}
