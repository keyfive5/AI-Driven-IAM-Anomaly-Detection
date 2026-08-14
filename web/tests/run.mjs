/** Node entry point for the test suite: `node web/tests/run.mjs`. */
import { runTests } from './tests.js';

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const DIM = '\x1b[2m';
const OFF = '\x1b[0m';

const { total, passed, failures } = await runTests((r) => {
  const mark = r.ok ? `${GREEN}✓${OFF}` : `${RED}✗${OFF}`;
  console.log(`${mark} ${r.name} ${DIM}${r.ms.toFixed(0)}ms${OFF}`);
  if (!r.ok) console.log(`  ${RED}${r.error}${OFF}`);
});

console.log(`\n${passed}/${total} passed`);
if (failures.length) {
  console.log(`${RED}${failures.length} failing${OFF}`);
  for (const f of failures) console.log(`  ${f.name}: ${f.error.message}`);
  process.exit(1);
}
