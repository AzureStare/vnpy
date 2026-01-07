import { useEffect, useRef } from "react";

export function useInterval(callback: () => void, delayMs: number | null) {
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    if (delayMs === null) return;
    const id = window.setInterval(() => cbRef.current(), delayMs);
    return () => window.clearInterval(id);
  }, [delayMs]);
}

