import { clearAuth, getToken } from "./auth";

export class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function _withCacheBust(url: string): string {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${Date.now()}`;
}

function _authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function fetchJson<T>(url: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(_withCacheBust(url), {
    ...init,
    headers: {
      ..._authHeaders(),
      ...(init.headers || {}),
    },
  });

  if (res.status === 401) {
    clearAuth();
    window.location.href = "/login.html";
    throw new ApiError("Unauthorized", 401);
  }

  if (!res.ok) {
    throw new ApiError(`${res.status} ${res.statusText}`, res.status);
  }

  return (await res.json()) as T;
}

export async function fetchText(url: string, init: RequestInit = {}): Promise<string> {
  const res = await fetch(_withCacheBust(url), {
    ...init,
    headers: {
      ..._authHeaders(),
      ...(init.headers || {}),
    },
  });

  if (res.status === 401) {
    clearAuth();
    window.location.href = "/login.html";
    throw new ApiError("Unauthorized", 401);
  }

  if (!res.ok) {
    throw new ApiError(`${res.status} ${res.statusText}`, res.status);
  }

  return await res.text();
}

