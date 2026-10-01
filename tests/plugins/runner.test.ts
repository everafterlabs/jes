import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { parseDecision, sessionOf } from "../../src/jes/agents/data/jes-runner.ts";

describe("jes runner", () => {
  it("reads a decision", () => {
    assert.deepEqual(parseDecision('{"ok": true, "onward": "hi", "decision": "allow"}\n'), {
      ok: true,
      onward: "hi",
      decision: "allow",
    });
    assert.deepEqual(parseDecision('{"ok": false, "onward": "Tool call blocked."}'), {
      ok: false,
      onward: "Tool call blocked.",
      decision: "block",
    });
  });

  it("blocks when jes prints nothing usable", () => {
    for (const stdout of [null, undefined, "", "  \n", "{", "[]", '{"ok": "yes", "onward": "x"}']) {
      const decision = parseDecision(stdout);
      assert.equal(decision.ok, false);
      assert.equal(decision.onward, "Blocked: jes did not answer.");
    }
  });

  it("cleans session ids", () => {
    assert.equal(sessionOf("sess/1", "x"), "sess-1");
    assert.equal(sessionOf(undefined, "pi"), "pi");
    assert.equal(sessionOf("-lead", "x"), "s-lead");
  });
});
