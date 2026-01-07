import fs from "node:fs";
import path from "node:path";

const assetsDir = path.resolve(process.cwd(), "../site/assets");

try {
  fs.rmSync(assetsDir, { recursive: true, force: true });
  // eslint-disable-next-line no-console
  console.log(`[prebuild] cleaned ${assetsDir}`);
} catch (err) {
  // eslint-disable-next-line no-console
  console.log(`[prebuild] failed to clean ${assetsDir}: ${String(err)}`);
}


