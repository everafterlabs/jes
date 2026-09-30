// Calls `uvx jes hook` on stdin. uvx ships with uv, so there is no separate jes install.
import { spawnSync } from "node:child_process";

export type JesDecision = { ok: boolean; onward: string; decision: string };

export type JesRunner = (body: Record<string, unknown>) => JesDecision;

export function spawnJes(body: Record<string, unknown>): JesDecision {
  const run = spawnSync("uvx", ["jes", "hook"], {
    input: JSON.stringify(body),
    encoding: "utf8",
  });
  const stdout = (run.stdout ?? "").trim();
  if (!stdout) {
    throw new Error("uvx jes hook failed");
  }
  return JSON.parse(stdout) as JesDecision;
}

export function sessionOf(value: string | undefined, fallback: string): string {
  const cleaned = (value ?? fallback).replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 200);
  return /^[A-Za-z0-9]/.test(cleaned) ? cleaned : `s${cleaned}`;
}
