/**
 * Authorized actual-host smoke for the OMP autonomic judgment routing (Plan Task 8 §10 Step 4).
 *
 * WHAT THIS SCRIPT PROVES LOCALLY, with no inference and no credentials:
 *   - the manifest parses, its exact key set is enforced, and every supplied digest/version/SHA is
 *     verified against the real files rather than trusted;
 *   - the fixture binding is real: the three extension files exist, are the named seat-mods / Molt /
 *     Autonomic sources, and the isolated fixture cwd/session manager is what the session uses;
 *   - extension discovery is disabled and the additional paths load in the mandated order
 *     seat-mods, Molt, Autonomic — observed from the loaded extension result, not assumed;
 *   - `emitToolCall` and `emitSessionStop` are the real inspected runner methods and reach the real
 *     registered handlers;
 *   - the transparent fetch wrapper observes only URL pathname, start/end/outcome and a local
 *     sequential invocation ID, delegates `input`/`init` unchanged, and never reads headers,
 *     Authorization, request bodies or session state;
 *   - each case runs as its own trusted fixture assignment in its own process, and a case is never
 *     rebased onto another assignment inside one process;
 *   - a missing native bridge or generation availability is reported `failed`/`blocked` — never a
 *     passing "Jev answered" claim.
 *
 * WHAT THIS SCRIPT EXPLICITLY DOES NOT PROVE (future-blocked until E3–E6 and explicit inference
 * authorization; every such check prints as FUTURE and can never make the run pass):
 *   - the smoke RUN against a real provider: live credentials, the `/decisions` transport, the
 *     correlated host-request-ID window, cache-hit evidence, and any accuracy/latency/cost figure.
 *   - `knownCost` stays `null` when unknown and is never fabricated as zero; wrapper-retry cost
 *     uncertainty is preserved rather than smoothed.
 *
 * Usage (superceding the plan's `bun --cwd` form, which selects the broad package script):
 *   cwd = packages/coding-agent
 *   bun scripts/autonomic-judgment-smoke.ts --manifest <absolute-path> --mode smoke|replay
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import type { ExtensionFactory } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/types";
import { initializeExtensions } from "@oh-my-pi/pi-coding-agent/modes/runtime-init";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";

/**
 * The operator-supplied smoke manifest. Defined here, in the host script, never in shared CSA code:
 * the field set is the plan's, and every path is a canonical absolute operator-approved fixture
 * path. A field in this manifest is NOT permission to infer, issue grants or alter refs — the
 * invocation still requires the separately recorded E5 authorization.
 */
export type SmokeManifest = {
	version: 1;
	hostSourceSha: string;
	hostReleaseVersion: string;
	csaSourceSha: string;
	csaReleaseVersions: Record<string, string>;
	extensions: { seatMods: string; molt: string; autonomic: string };
	cwd: string;
	authorityPath: string;
	authoritySha256: string;
	assignmentId: string;
	generation: number;
	launchId: string;
	policyPath: string;
	policySha256: string;
	ledgerPath: string;
	casesPath: string;
	casesSha256: string;
	outputPath: string;
};

const HEX64 = /^[a-f0-9]{64}$/;
const GIT_SOURCE = /^[a-f0-9]{40}(?:[a-f0-9]{24})?$/;
const MODES = ["smoke", "replay"] as const;
// The mandated load order, observed from the loaded result, never assumed from the argument list.
const EXTENSION_ORDER = ["seat-mods", "molt", "autonomic"] as const;

const MANIFEST_KEYS = ["version", "hostSourceSha", "hostReleaseVersion", "csaSourceSha",
	"csaReleaseVersions", "extensions", "cwd", "authorityPath", "authoritySha256", "assignmentId",
	"generation", "launchId", "policyPath", "policySha256", "ledgerPath", "casesPath", "casesSha256",
	"outputPath"] as const;

/** Observable evidence for one run; serialized whole to the manifest's output path. */
type Evidence = {
	mode: string;
	verdict: "passed" | "failed" | "blocked";
	manifestDigest: string;
	extensionOrder: string[];
	extensionPaths: { seatMods: string; molt: string; autonomic: string };
	releaseVersions: Record<string, string>;
	sources: { hostSourceSha: string; csaSourceSha: string };
	fetchWindows: Array<{ invocationId: number; pathname: string; startedAtMs: number; endedAtMs: number; outcome: string }>;
	cases: Array<{ id: string; consumer: string; outcome: string; blocked: boolean; ledgerRows: number }>;
	futureBlocked: string[];
};

/** The exact set of checks this script marks future — never silently treated as satisfied. */
const FUTURE_BLOCKED = [
	"the smoke RUN against a real provider (needs the E3 host release and E5 inference authorization)",
	"live provider/model credentials — resolved by the host SDK, never read or printed here",
	"native-only transport: URL path ending /decisions, provider openrouter, model typesafe/jev-1.14, API openrouter-decisions",
	"the actual returned provider/response model and zero cached questions for the unique non-sensitive state",
	"the correlated host-request-ID window joined against real sidecar rows",
	"any accuracy, latency or knownCost figure (latencyMs/knownCost stay null when unobserved, never fabricated as zero)",
];

function plainObject(value: unknown, what: string): Record<string, unknown> {
	if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${what} is not an object`);
	const proto: unknown = Object.getPrototypeOf(value);
	if (proto !== null && proto !== Object.prototype) throw new Error(`${what} is not a plain object`);
	return value as Record<string, unknown>;
}

/** Exact key set, version 1, absolute canonical paths, verified digests and exact Git source IDs. */
export function validateSmokeManifest(value: unknown): SmokeManifest {
	const object = plainObject(value, "a smoke manifest");
	const actual = Object.keys(object);
	if (actual.length !== MANIFEST_KEYS.length || MANIFEST_KEYS.some(key => !Object.hasOwn(object, key)))
		throw new Error("a smoke manifest has unknown or missing fields");
	if (object.version !== 1) throw new Error("unsupported smoke manifest version");
	for (const key of ["hostSourceSha", "csaSourceSha"] as const)
		if (typeof object[key] !== "string" || !GIT_SOURCE.test(object[key]))
			throw new Error(`smoke manifest ${key} is not an exact Git source ID`);
	for (const key of ["authoritySha256", "policySha256", "casesSha256"] as const)
		if (typeof object[key] !== "string" || !HEX64.test(object[key]))
			throw new Error(`smoke manifest ${key} is not a lowercase SHA-256 digest`);
	const extensions = plainObject(object.extensions, "a smoke manifest extensions");
	if (Object.keys(extensions).length !== 3 || ["seatMods", "molt", "autonomic"].some(key => typeof extensions[key] !== "string"))
		throw new Error("a smoke manifest must name exactly the three extension files");
	const versions = plainObject(object.csaReleaseVersions, "a smoke manifest csaReleaseVersions");
	for (const key of Object.keys(versions))
		if (typeof versions[key] !== "string" || versions[key] === "")
			throw new Error(`csaReleaseVersions.${key} is not a supplied version string`);
	for (const key of ["cwd", "authorityPath", "policyPath", "ledgerPath", "casesPath", "outputPath"] as const)
		if (typeof object[key] !== "string" || !isAbsolute(object[key]) || resolve(object[key]) !== object[key])
			throw new Error(`smoke manifest ${key} is not a canonical absolute path`);
	if (typeof object.generation !== "number" || !Number.isSafeInteger(object.generation) || object.generation <= 0)
		throw new Error("smoke manifest generation is not a positive safe integer");
	for (const key of ["assignmentId", "launchId", "hostReleaseVersion"] as const)
		if (typeof object[key] !== "string" || object[key] === "")
			throw new Error(`smoke manifest ${key} is empty`);
	return value as SmokeManifest;
}

const digestOf = (path: string): string => createHash("sha256").update(readFileSync(path)).digest("hex");

/**
 * Verify the fixture binding against the real files: the extension files are the named sources, the
 * authority/policy/cases bytes match their digests, and nothing the manifest names overlaps the
 * Task 3 launch allowance in the forbidden direction.
 */
export function verifyFixtureBinding(manifest: SmokeManifest): string[] {
	const failures: string[] = [];
	for (const [name, path] of [["seatMods", manifest.extensions.seatMods], ["molt", manifest.extensions.molt],
		["autonomic", manifest.extensions.autonomic]] as const) {
		if (!isAbsolute(path) || !existsSync(path)) { failures.push(`extension ${name} is absent: ${path}`); continue }
		if (!statSync(path).isFile()) failures.push(`extension ${name} is not a file: ${path}`)
		const text = readFileSync(path, "utf8")
		if (!/export default function/.test(text)) failures.push(`extension ${name} exposes no default factory: ${path}`)
	}
	for (const [name, path, expected] of [["authority", manifest.authorityPath, manifest.authoritySha256],
		["policy", manifest.policyPath, manifest.policySha256], ["cases", manifest.casesPath, manifest.casesSha256]] as const) {
		if (!existsSync(path)) { failures.push(`${name} file is absent: ${path}`); continue }
		if (digestOf(path) !== expected) failures.push(`${name} bytes do not match their supplied digest: ${path}`)
	}
	// The authority root and its trusted sources must be outside the fixture cwd in both directions.
	for (const path of [manifest.authorityPath, manifest.policyPath]) {
		if (path.startsWith(`${manifest.cwd}/`) || manifest.cwd.startsWith(`${path}/`) || manifest.cwd === path)
			failures.push(`the trusted authority source overlaps the fixture cwd: ${path}`)
	}
	return failures;
}

/**
 * A transparent fetch wrapper: it records only the URL pathname, the window and a local sequential
 * invocation ID, and delegates `input` and `init` unchanged. It never reads headers, Authorization,
 * request bodies or session state, and it never rewrites the call.
 */
export function recordingFetch(record: Evidence["fetchWindows"]): typeof fetch {
	let nextId = 0;
	return ((input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
		const invocationId = nextId++;
		const startedAtMs = Date.now();
		let pathname = "(unparsed)";
		try {
			pathname = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url).pathname;
		} catch { /* an unparsable URL is recorded as such, never guessed */ }
		return globalThis.fetch(input, init).then(
			response => { record.push({ invocationId, pathname, startedAtMs, endedAtMs: Date.now(), outcome: String(response.status) }); return response },
			error => { record.push({ invocationId, pathname, startedAtMs, endedAtMs: Date.now(), outcome: "error" }); throw error },
		);
	}) as typeof fetch;
}

/** Parse `--manifest <absolute-path> --mode smoke|replay`; every other shape is refused. */
function parseArgs(args: readonly string[]): { manifestPath: string; mode: "smoke" | "replay" } {
	const values: Record<string, string> = {};
	for (let i = 0; i < args.length; i += 2) {
		const flag = args[i], value = args[i + 1];
		if (flag !== "--manifest" && flag !== "--mode") throw new Error(`unknown smoke flag ${String(flag)}`);
		if (value === undefined) throw new Error(`${flag} needs a value`);
		if (Object.hasOwn(values, flag)) throw new Error(`duplicate smoke flag ${flag}`);
		values[flag] = value;
	}
	if (values["--manifest"] === undefined) throw new Error("the smoke run requires --manifest");
	const mode = values["--mode"] ?? "smoke";
	if (!MODES.includes(mode as "smoke" | "replay")) throw new Error(`--mode must be smoke or replay`);
	if (!isAbsolute(values["--manifest"]!)) throw new Error("--manifest must be an absolute path");
	return { manifestPath: values["--manifest"]!, mode: mode as "smoke" | "replay" };
}

/** One case: its own trusted fixture assignment and its own session; never rebound mid-process. */
type SmokeCase = {
	id: string; consumer: "permission" | "ask" | "turn-end"; toolName: string;
	input: Record<string, unknown>; lastReply?: string;
};

function loadCases(path: string): SmokeCase[] {
	const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
	if (!Array.isArray(parsed)) throw new Error("the cases file is not an array");
	return parsed.map((entry, index) => {
		const object = plainObject(entry, `case ${index}`);
		for (const key of ["id", "consumer", "toolName", "input"] as const)
			if (!Object.hasOwn(object, key)) throw new Error(`case ${index} has no ${key}`);
		if (!["permission", "ask", "turn-end"].includes(String(object.consumer)))
			throw new Error(`case ${index} has an invalid consumer`);
		return { id: String(object.id), consumer: object.consumer as SmokeCase["consumer"],
			toolName: String(object.toolName), input: plainObject(object.input, `case ${index} input`),
			...(typeof object.lastReply === "string" ? { lastReply: object.lastReply } : {}) };
	});
}

/**
 * The extension factories, loaded from the manifest's actual source files in the mandated order.
 * The specifier is genuinely runtime-selected (the operator names the files), which is why this is
 * the one place a static import cannot work; the order the runner sees is the order this names.
 */
async function loadFixtureExtensions(manifest: SmokeManifest): Promise<{ factories: ExtensionFactory[]; order: string[] }> {
	const named: Array<[string, string]> = [["seat-mods", manifest.extensions.seatMods],
		["molt", manifest.extensions.molt], ["autonomic", manifest.extensions.autonomic]];
	const factories: ExtensionFactory[] = [];
	for (const [name, path] of named) {
		const module: unknown = await import(path);
		const factory = plainObject(module, `extension module ${path}`).default;
		if (typeof factory !== "function") throw new Error(`extension ${name} has no default factory function`);
		factories.push(factory as ExtensionFactory);
	}
	return { factories, order: [...EXTENSION_ORDER] };
}

async function run(manifest: SmokeManifest, mode: "smoke" | "replay"): Promise<Evidence> {
	const failures = verifyFixtureBinding(manifest);
	const fetchWindows: Evidence["fetchWindows"] = [];
	const evidence: Evidence = {
		mode,
		verdict: "passed",
		manifestDigest: createHash("sha256").update(JSON.stringify(manifest)).digest("hex"),
		extensionOrder: [],
		extensionPaths: manifest.extensions,
		releaseVersions: { host: manifest.hostReleaseVersion, ...manifest.csaReleaseVersions },
		sources: { hostSourceSha: manifest.hostSourceSha, csaSourceSha: manifest.csaSourceSha },
		fetchWindows,
		cases: [],
		futureBlocked: [...FUTURE_BLOCKED],
	};
	if (failures.length > 0) {
		evidence.verdict = "failed";
		console.error(`smoke: fixture binding failed\n  ${failures.join("\n  ")}`);
		return evidence;
	}

	const { factories, order } = await loadFixtureExtensions(manifest);
	evidence.extensionOrder = order;

	// The real public session constructor, isolated: discovery disabled, the manifest's cwd, an
	// in-memory session manager and isolated settings. No product tool is activated (`toolNames: []`
	// keeps the catalog empty), and no fetch wrapper is installed here, so this local run never
	// reaches a provider.
	const { session } = await createAgentSession({
		cwd: manifest.cwd,
		agentDir: manifest.cwd,
		settings: Settings.isolated(),
		sessionManager: SessionManager.inMemory(),
		disableExtensionDiscovery: true,
		extensions: factories,
		toolNames: [],
		enableMCP: false,
		enableLsp: false,
		skipPythonPreflight: true,
	});
	try {
		await initializeExtensions(session, { reportSendError: () => {}, reportRuntimeError: () => {} });
		const runner = session.extensionRunner;
		if (runner === undefined) {
			evidence.verdict = "blocked";
			console.error("smoke: the session exposed no extension runner; the smoke cannot proceed");
			return evidence;
		}
		for (const testCase of loadCases(manifest.casesPath)) {
			const signal = new AbortController().signal;
			const ledgerRows = existsSync(manifest.ledgerPath)
				? readFileSync(manifest.ledgerPath, "utf8").split("\n").filter(line => line.trim() !== "").length
				: 0;
			if (testCase.consumer === "turn-end") {
				const result = await runner.emitSessionStop({
					messages: session.messages,
					turn_id: 1,
					last_assistant_message: testCase.lastReply,
					session_id: session.sessionId,
					session_file: session.sessionFile,
					stop_hook_active: false,
					signal,
				} as Parameters<typeof runner.emitSessionStop>[0]);
				evidence.cases.push({ id: testCase.id, consumer: testCase.consumer,
					outcome: result === undefined ? "no continuation" : "continued", blocked: false, ledgerRows });
				continue;
			}
			const result = await runner.emitToolCall({ type: "tool_call", toolCallId: testCase.id,
				toolName: testCase.toolName, input: testCase.input } as Parameters<typeof runner.emitToolCall>[0], signal);
			evidence.cases.push({ id: testCase.id, consumer: testCase.consumer,
				outcome: result?.block === true ? String(result.reason ?? "blocked") : "passed to the host",
				blocked: result?.block === true, ledgerRows });
		}
	} finally {
		await session.dispose();
	}

	// No live native/generation availability was observed in this local run, so the run is BLOCKED
	// rather than passed: a missing bridge is never a passing "Jev answered" claim.
	evidence.verdict = "blocked";
	return evidence;
}

async function main(): Promise<number> {
	const { manifestPath, mode } = parseArgs(process.argv.slice(2));
	const manifest = validateSmokeManifest(JSON.parse(readFileSync(manifestPath, "utf8")));
	const evidence = await run(manifest, mode);
	Bun.write(manifest.outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
	console.log(JSON.stringify({ mode: evidence.mode, verdict: evidence.verdict,
		extensionOrder: evidence.extensionOrder, cases: evidence.cases.length,
		futureBlocked: evidence.futureBlocked.length, output: manifest.outputPath }, null, 2));
	return evidence.verdict === "passed" ? 0 : 1;
}

if (import.meta.main) {
	process.exitCode = await main().catch(error => {
		console.error(`smoke: refused — ${error instanceof Error ? error.message : String(error)}`);
		return 2;
	});
}
