/** Same-origin transport. Tokens stay out of query strings and server logs. */
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";

export async function api(path, {method = "GET", body, signal, image = false, timeoutMs = 120000} = {}) {
  if (!Number.isFinite(timeoutMs) || timeoutMs < 0) throw new TypeError("Invalid request timeout");
  const headers = {"X-Viewer-Token": token};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const controller = new AbortController();
  let timedOut = false;
  const forwardAbort = () => controller.abort(signal.reason);
  if (signal?.aborted) forwardAbort();
  else signal?.addEventListener("abort", forwardAbort, {once: true});
  const timer = timeoutMs > 0 ? setTimeout(() => {
    timedOut = true; controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, timeoutMs) : null;
  try {
    const response = await fetch(path, {
      method, headers, signal: controller.signal, cache: "no-store", credentials: "same-origin",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      let message = `Local service returned ${response.status}`;
      try { message = (await response.json()).error || message; } catch { /* Not a JSON response. */ }
      if (controller.signal.aborted) throw controller.signal.reason;
      if ([401, 403].includes(response.status) && !image) message += ". Open the complete URL printed by the server, including #token=…";
      const error = new Error(message);
      error.status = response.status;
      throw error;
    }
    return await (image ? response.blob() : response.json());
  } catch (error) {
    if (timedOut) {
      const timeout = new Error("Local service timed out. Refresh the operation status.");
      timeout.name = "TimeoutError"; timeout.status = 408; throw timeout;
    }
    throw error;
  } finally {
    if (timer !== null) clearTimeout(timer);
    signal?.removeEventListener("abort", forwardAbort);
  }
}
