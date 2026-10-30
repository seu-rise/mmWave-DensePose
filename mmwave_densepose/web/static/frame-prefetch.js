/** A bounded, generation-scoped queue of decoded frames, used only in playback.
 * take() transfers ownership to the caller. All other entries belong to this
 * queue and are released on eviction, pause, seek, sequence change or teardown.
 */
export class FramePrefetch {
  constructor(load, {maxFrames = 4, maxBytes = 48 * 1024 ** 2, concurrency = 2} = {}) {
    this.load = load;
    this.maxFrames = maxFrames; this.maxBytes = maxBytes; this.concurrency = concurrency;
    this.entries = new Map(); this.jobs = new Set(); this.active = false;
  }

  begin(generation, next, last, frameBytes) {
    this.clear();
    this.active = true; this.generation = generation;
    this.next = next; this.last = last;
    this.estimate = Math.max(1, frameBytes || 12 * 1024 ** 2);
    this.fill();
  }

  clear() {
    this.active = false;
    for (const entry of this.entries.values()) this.drop(entry);
    this.entries.clear();
  }

  async warm() {
    // A short initial buffer absorbs the first request/decode latency too.
    await Promise.allSettled([...this.entries.values()].slice(0, this.concurrency).map(entry => entry.promise));
  }

  drop(entry) {
    this.entries.delete(entry.index);
    entry.controller.abort();
    entry.value?.release(); entry.value = null;
    entry.reject(new DOMException("Frame prefetch cancelled", "AbortError"));
  }

  fill() {
    if (!this.active) return;
    let used = [...this.entries.values()].reduce((sum, entry) => sum + entry.bytes, 0);
    for (let index = this.next; index <= Math.min(this.last, this.next + this.maxFrames - 1); index++) {
      if (this.entries.has(index)) continue;
      if (this.entries.size >= this.maxFrames || used + this.estimate > this.maxBytes) break;
      let resolve, reject;
      const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
      // Background failures are observed here; take() still receives the error.
      promise.catch(() => {});
      this.entries.set(index, {index, generation: this.generation, controller: new AbortController(),
        status: "queued", bytes: this.estimate, promise, resolve, reject, value: null});
      used += this.estimate;
    }
    this.pump();
  }

  pump() {
    if (!this.active) return;
    for (const entry of this.entries.values()) {
      if (this.jobs.size >= this.concurrency) break;
      if (entry.status !== "queued") continue;
      entry.status = "loading"; this.jobs.add(entry);
      this.load(entry.index, entry.generation, entry.controller.signal).then(value => {
        if (entry.controller.signal.aborted) { value.release(); return; }
        entry.value = value; entry.bytes = value.bytes; entry.status = "ready";
        // Larger-than-expected frames evict furthest lookahead entries first.
        let used = [...this.entries.values()].reduce((sum, item) => sum + item.bytes, 0);
        for (const item of [...this.entries.values()].reverse()) {
          if (used <= this.maxBytes) break;
          if (item === entry || item.index <= this.next) continue;
          used -= item.bytes; this.drop(item);
        }
        entry.resolve(value);
      }, error => { entry.status = "failed"; entry.bytes = 0; entry.reject(error); })
        .finally(() => { this.jobs.delete(entry); this.pump(); });
    }
  }

  async take(index, generation, signal) {
    const entry = generation === this.generation ? this.entries.get(index) : null;
    if (!entry) return this.load(index, generation, signal);
    const abort = () => this.drop(entry);
    signal.addEventListener("abort", abort, {once: true});
    try {
      if (signal.aborted) abort();
      const value = await entry.promise;
      if (signal.aborted) throw new DOMException("Frame request cancelled", "AbortError");
      this.entries.delete(index); entry.value = null; // ownership moves to caller
      this.next = index + 1;
      this.estimate = Math.max(1, value.bytes);
      this.fill();
      return value;
    } finally { signal.removeEventListener("abort", abort); }
  }

  advance(index, generation, bytes) {
    if (!this.active || generation !== this.generation) return;
    this.next = index + 1; this.estimate = Math.max(1, bytes);
    for (const entry of [...this.entries.values()]) if (entry.index < this.next) this.drop(entry);
    this.fill();
  }
}
