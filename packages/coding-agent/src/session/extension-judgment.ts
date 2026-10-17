/**
 * Session-bound judgment bridge for extensions.
 *
 * An extension handler asks its own session for a native-only judgment; the
 * session supplies settings, registry, identity, cache and accounting, so the
 * request cannot be routed by, or billed to, another session. The request is
 * captured as immutable JSON before any credential or promise work, the whole
 * judgment is raced against one total deadline, and every failure surfaces as
 * an opaque error that never leaks provider text, headers or state.
 *
 * The extension path is the invocation owner's, taken from the live runner
 * scope — never from the request or the caller-supplied `options.purpose`.
 */
import { createHash } from "node:crypto";
import type { JudgmentRequest, JudgmentResult, JsonValue, Questions } from "@oh-my-pi/pi-ai";
import { type JudgeDeps, type JudgmentRoute, resolveJudge } from "../judgment";

/** Caller-controlled options for one extension judgment. */
export type ExtensionJudgmentOptions = {
	/** Fixed role purpose; the bridge namespaces it with the owning extension and request id. */
	purpose: string;
	/** Extensions may only reach native System One judgments. */
	nativeOnly: true;
	/** Total end-to-end budget in milliseconds. */
	timeoutMs: number;
	/** Optional caller cancellation, combined with the live invocation scope. */
	signal?: AbortSignal;
};

/** One complete judgment: the result plus the route that actually served it. */
export type ExtensionJudgment<Q extends Questions> = {
	judgment: JudgmentResult<Q>;
	route: {
		requestId: string;
		purpose: string;
		selected: { api: string; provider: string; model: string };
		questions: number;
		cachedQuestions: number;
	};
};

/** Session-bound judge exposed as `ctx.judge`. */
export type ExtensionJudge = <Q extends Questions>(
	request: JudgmentRequest<Q>,
	options: ExtensionJudgmentOptions,
) => Promise<ExtensionJudgment<Q>>;

/** Host action that accepts the owning extension path as its first argument. */
export type ExtensionJudgeAction = <Q extends Questions>(
	extensionPath: string,
	request: JudgmentRequest<Q>,
	options: ExtensionJudgmentOptions,
) => Promise<ExtensionJudgment<Q>>;

/**
 * Recursively copy JSON state into frozen containers. Every container is
 * checked through its property descriptors, so an accessor or `toJSON` is
 * rejected rather than invoked — the provider must receive the bytes the
 * caller wrote, not whatever a getter decided to return by request time.
 *
 * `stack` tracks the current recursion path, not every seen object, so a
 * subobject shared by two siblings is copied twice instead of being mistaken
 * for a cycle.
 */
export function captureJson(value: unknown): JsonValue {
	return copyJson(value, new Set<object>());
}

function copyJson(value: unknown, stack: Set<object>): JsonValue {
	if (value === null) return null;
	switch (typeof value) {
		case "boolean":
		case "string":
			return value;
		case "number": {
			if (!Number.isFinite(value)) throw new Error("judgment state must be finite JSON");
			return value;
		}
		case "object":
			break;
		default:
			// undefined, symbol, bigint, function — none are JSON.
			throw new Error("judgment state must be JSON");
	}
	const source = value as object;
	if (stack.has(source)) throw new Error("judgment state must not contain cycles");
	stack.add(source);
	try {
		return Array.isArray(source) ? copyArray(source, stack) : copyObject(source, stack);
	} finally {
		stack.delete(source);
	}
}

function copyArray(source: readonly unknown[], stack: Set<object>): readonly JsonValue[] {
	const lengthDescriptor = Object.getOwnPropertyDescriptor(source, "length");
	if (!lengthDescriptor || !("value" in lengthDescriptor)) throw new Error("judgment state array is not plain");
	const length = lengthDescriptor.value as number;
	const keys = Object.keys(source);
	// A hole (length differs from the enumerable own indices) is not JSON.
	if (keys.length !== length) throw new Error("judgment state must not contain sparse arrays");
	const copy: JsonValue[] = [];
	for (let index = 0; index < length; index += 1) {
		copy.push(copyOwn(source, String(index), stack));
	}
	return Object.freeze(copy);
}

function copyObject(source: object, stack: Set<object>): { readonly [key: string]: JsonValue } {
	const prototype = Object.getPrototypeOf(source) as object | null;
	if (prototype !== Object.prototype && prototype !== null) {
		throw new Error("judgment state must contain only plain objects");
	}
	const copy: Record<string, JsonValue> = {};
	for (const key of Reflect.ownKeys(source)) {
		if (typeof key !== "string") throw new Error("judgment state must not contain symbol keys");
		copy[key] = copyOwn(source, key, stack);
	}
	return Object.freeze(copy);
}

/** Read one own property through its descriptor: data values are copied, accessors are refused. */
function copyOwn(source: object, key: string, stack: Set<object>): JsonValue {
	const descriptor = Object.getOwnPropertyDescriptor(source, key);
	if (!descriptor) throw new Error("judgment state must contain only own properties");
	if (!("value" in descriptor)) throw new Error("judgment state must not contain accessors");
	if (!descriptor.enumerable) throw new Error("judgment state must not contain non-enumerable keys");
	return copyJson(descriptor.value, stack);
}

/**
 * Run one operation under a single total deadline. The caller's own signals
 * are combined with this budget; when either fires the provider work sees the
 * composed signal, so a cancel is a real cancellation and not merely a race.
 */
export async function settleWithin<T>(
	run: (signal: AbortSignal) => Promise<T>,
	timeoutMs: number,
	inherited: readonly AbortSignal[],
): Promise<T> {
	if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("judgment unavailable: no budget");
	const deadline = performance.now() + timeoutMs;
	const controller = new AbortController();
	const signal = AbortSignal.any([...inherited, controller.signal]);
	let timer: ReturnType<typeof setTimeout> | undefined;
	let onAbort: (() => void) | undefined;
	try {
		signal.throwIfAborted();
		const stopped = new Promise<never>((_resolve, reject) => {
			onAbort = () => reject(new Error("judgment cancelled"));
			signal.addEventListener("abort", onAbort, { once: true });
			timer = setTimeout(() => controller.abort(), timeoutMs);
		});
		const result = await Promise.race([run(signal), stopped]);
		signal.throwIfAborted();
		if (performance.now() >= deadline) throw new Error("judgment deadline");
		return result;
	} finally {
		if (timer !== undefined) clearTimeout(timer);
		if (onAbort) signal.removeEventListener("abort", onAbort);
		controller.abort();
	}
}

/**
 * One native-only judgment for the owning extension. The request is frozen
 * before any promise is created or credential resolved; the purpose is
 * namespaced with the owner hash and a request id so existing usage rows and
 * telemetry spans correlate every wrapper attempt without a new schema.
 */
export async function runExtensionJudgment<Q extends Questions>(
	deps: JudgeDeps,
	request: JudgmentRequest<Q>,
	options: ExtensionJudgmentOptions,
	live: { signal: AbortSignal; current: () => boolean },
	extensionPath: string,
): Promise<ExtensionJudgment<Q>> {
	const requestId = crypto.randomUUID();
	// The owner comes from the runner's invocation path, never from the request or options.
	const owner = createHash("sha256").update(extensionPath).digest("hex");
	const purpose = `extension:${owner}:${options.purpose}:${requestId}`;
	let route: JudgmentRoute | undefined;
	try {
		const frozen = captureJson(request) as unknown as JudgmentRequest<Q>;
		if (options.nativeOnly !== true || !options.purpose.trim() || !live.current()) throw new Error("unavailable");
		const judgment = await settleWithin(
			signal =>
				resolveJudge({
					...deps,
					purpose,
					nativeOnly: true,
					onResult: value => {
						route = value;
					},
				}).judge(frozen, { signal }),
			options.timeoutMs,
			[live.signal, ...(options.signal ? [options.signal] : [])],
		);
		if (!live.current() || !route) throw new Error("stale or unavailable");
		return { judgment, route: { requestId, purpose, ...route } };
	} catch {
		// Never expose registry/provider error text, headers, state, or a fabricated result.
		throw new Error(`extension judgment ${requestId}: unavailable or cancelled`);
	}
}
