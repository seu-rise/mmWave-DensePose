// Stored timestamps are Unix milliseconds; format only the display, never the data.
const pad = (value, width = 2) => String(value).padStart(width, "0");
const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;

export function formatTimestamp(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "not stored";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "not stored";
  return `${pad(date.getFullYear(), 4)}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} `
    + `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`;
}

export function showTimestamp(element, value, prefix = "") {
  const formatted = formatTimestamp(value);
  element.textContent = prefix + formatted;
  if (formatted === "not stored") element.removeAttribute("title");
  else element.title = `${value} ms · ${timeZone}`;
}
