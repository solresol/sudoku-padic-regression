# Browser LanguageModel CNF Workflow

This document records the workflow used by the CSP page and by `npm run test:language-model`.

## Current Prompting Sequence

There is no system prompt.

1. Record `LanguageModel.availability()` in the visible debug transcript.
2. Start a fresh `LanguageModel` session for schema extraction. Session creation uses Chrome's `downloadprogress` monitor when the browser exposes it, and those events are recorded in the transcript with role `browser`.
3. Ask for a typed assignment schema, not CNF clauses:

```text
Extract a typed JSON schema for this assignment CSP. Output JSON only.
Do not output CNF clauses. Do not explain. Do not use Markdown.
Schema shape:
{"kind":"assignment","people":["Ava"],"jobs":["test"],"person_exactly_one_job":true,"job_exactly_one_person":true,"forbidden":[{"person":"Ava","job":"test","source":"Ava cannot test."}],"implications":[{"if":{"person":"Devina","job":"documentation"},"then":{"person":"Ava","job":"design"},"source":"If Devina documents, then Ava designs."}]}
Rules:
- people and jobs must contain only names stated by the problem.
- For `Neither Ben nor Cara can do documentation`, output two forbidden entries, one for Ben and one for Cara.
- For `If X does job1, then Y does job2`, output one implication with `if` and `then` literals.
- Set both exactly-one booleans only when the problem says everyone gets exactly one job and every job gets exactly one person.
--- Problem ---
[problem]
```

4. The app cleans non-JSON wrapping internally, then shows user-facing CNF progress while keeping the raw model conversation hidden.
5. The app deterministically expands the schema into CNF:
   - `person_exactly_one_job` creates one at-least-one clause per person and all pairwise at-most-one clauses for that person;
   - `job_exactly_one_person` creates one at-least-one clause per job and all pairwise at-most-one clauses for that job;
   - every `forbidden` entry becomes a unit clause;
   - every `implication` entry becomes `not antecedent or consequent`.
6. The generated clauses are compiled and checked before they are displayed.

## Default Test Problem

```text
Ava, Ben, Cara, and Devina need to be assigned test, design, documentation and development. Everyone needs exactly one job. Every job needs exactly one person assigned to it.
Rules: Ava cannot test. Neither Ben nor Cara can do  documentation. If Devina documents, then Ava designs.
```

The in-memory cache has the expected 60 CNF clauses for this problem, but cache use is disabled while the workflow is being debugged.

Critical clauses expected from the language model:

```text
not Ava_test
not Ben_documentation
not Cara_documentation
not Devina_documentation or Ava_design
```

The old rule `Whoever does documentation can't do development` is no longer part of the default problem. The model should not introduce clauses for that separate rule.

```text
Whoever does documentation can't do development.
```

Important distinction: clauses like `not Ava_documentation or not Ava_development` can still be valid when their purpose is `Ava must have exactly one job`; they are not valid if their purpose is the removed documentation/development rule.

## Chrome for Testing Automation

Run:

```bash
npm run test:language-model
```

The script launches:

```text
/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing
```

It uses a temporary profile at:

```text
/tmp/padic-logic-cft-profile
```

By default it tests:

```text
https://padic-logic.symmachus.org/
```

Useful environment variables:

```bash
LANGUAGE_MODEL_TEST_URL=http://127.0.0.1:5173/ npm run test:language-model
LANGUAGE_MODEL_WAIT_MS=1800000 npm run test:language-model
LANGUAGE_MODEL_API_WAIT_MS=30000 npm run test:language-model
LANGUAGE_MODEL_PROGRESS_LOG_MS=30000 npm run test:language-model
CFT_PROFILE_DIR=/tmp/another-profile npm run test:language-model
CHROME_FOR_TESTING_PATH="/Applications/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing" npm run test:language-model
```

The harness does not wait for `LanguageModel.availability()` to become `available` before clicking the app. Chrome reports `downloadable` or `downloading` before the first origin use, and the documented trigger path is `LanguageModel.create()` after user activation. The harness therefore verifies that the API is present and not `unavailable`, fills the problem textarea, and clicks `Generate CSP` with Chrome DevTools Protocol mouse events.

The default generation wait is 30 minutes. While waiting, the harness polls the rendered page from Node and logs progress every 30 seconds: elapsed time, whether the app is still generating, how many source lines exist, and the latest visible transcript message. This is deliberately patient because the first Chrome on-device model download can be slow.

The result JSON is written to:

```text
tmp/language-model-cft-result.json
```

Exit codes:

- `0`: all LanguageModel cases passed.
- `1`: LanguageModel ran but produced wrong or incomplete CNF.
- `2`: Chrome for Testing launched, but the `LanguageModel` API was missing or `unavailable`.

## Codex Chrome Extension Testing

The Codex Chrome extension can test the already-running user Chrome profile. This is useful when Chrome for Testing is stuck downloading the on-device model, because the user's normal Chrome profile may already have the model available.

Observed on 2026-07-09:

1. Claiming an older open `p-adic logic` tab showed stale behavior from a pre-deployment bundle: the transcript still contained the removed system prompt and the model returned `The input is too large.`
2. Opening a fresh tab in the same Chrome profile loaded the current bundle and the app reported:

```text
LanguageModel.availability(): available
Language model download progress 100%.
Local language model session ready for variable extraction.
```

3. The local model reached real CNF generation, which exposed validator gaps:

```text
not Ben_Documentation or not Cara_Documentation
Ava_Test
```

The first clause is too weak for `Neither Ben nor Cara can do documentation`; it must be two separate unit clauses. The second contradicted the already accepted `not Ava_Test` clause. The validator now rejects both patterns.

## Test Cases

The automation currently runs:

1. `default-assignment`: the four-person/four-job problem above. This case must produce 60 clauses.
2. `tiny-assignment`: two people, two jobs, and one negative rule.
3. `single-implication`: only `If Devina documents, then Ava designs.`

The tiny and implication cases are deliberately smaller than the default problem. If the model fails the default case, these isolate whether it is failing exactly-one expansion, implication conversion, or basic JSON/CNF output.

## Current Simplification Strategy

The model no longer enumerates CNF clauses. It only extracts a small schema. The deterministic compiler handles the high-cardinality pieces that the model handled poorly: exactly-one expansion, separate unit clauses for `neither/nor`, and implication conversion.

## Observed Chrome for Testing Result

On 2026-07-08, an early short-timeout run used this command:

```bash
CFT_PROFILE_DIR="$HOME/Library/Application Support/Google/Chrome for Testing" LANGUAGE_MODEL_TEST_URL=http://127.0.0.1:5173/ LANGUAGE_MODEL_WAIT_MS=60000 npm run test:language-model
```

launched Chrome for Testing successfully and found `window.LanguageModel`, but Chrome reported:

```text
LanguageModel.availability(): downloading
Creating local language model session for variable extraction.
```

The page then stayed on `Generating...` until the harness timed out. The generated CSP source remained blank, so that run was blocked on Chrome's local model download/session creation before the variable prompt was sent.

On 2026-07-09, the same persistent Chrome for Testing profile was given a 15-minute wait. It still timed out before the variable prompt was sent:

```text
elapsedMs: 900809
LanguageModel.availability(): downloading
Creating local language model session for variable extraction.
Language model download progress 0%.
```

That run suggests the browser entered the model download path but did not receive model bytes during the 15-minute window. The harness now defaults to 30 minutes and reports progress while it waits.

## References

- Chrome Prompt API documentation: https://developer.chrome.com/docs/ai/prompt-api
- Chrome model download guidance: https://developer.chrome.com/docs/ai/inform-users-of-model-download
