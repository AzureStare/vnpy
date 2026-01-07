export type UserRole = "admin" | "user" | "unknown";

const TOKEN_KEY = "flagship_token";
const ROLE_KEY = "flagship_role";

export function getToken(): string | null {
  const t = localStorage.getItem(TOKEN_KEY);
  return t && t.trim() ? t : null;
}

export function getRole(): UserRole {
  const r = (localStorage.getItem(ROLE_KEY) || "").trim().toLowerCase();
  if (r === "admin" || r === "user") return r;
  return "unknown";
}

export function isAdminRole(role: UserRole): boolean {
  return role === "admin";
}

export function clearAuth(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(ROLE_KEY);
  } catch (_) {
    // ignore
  }
}

