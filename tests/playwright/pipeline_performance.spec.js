const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const ARTIFACT_DIR = path.resolve('.webui_cache', 'playwright_artifacts');
const TIMING_PATH = path.join(ARTIFACT_DIR, 'pipeline_timings.json');
const SECOND_SCREENSHOT = path.join(ARTIFACT_DIR, 'second-set-loaded.png');
const THIRD_SCREENSHOT = path.join(ARTIFACT_DIR, 'third-set-loaded.png');

const STEP_STAGE_SPECS = [
  ['Auto-orient parts', 5000],
  ['Auto-stack parts', 5000],
  ['Auto-scale parts', 5000],
  ['Autodrop', 5000],
  ['Export parts', 5000],
  ['Render assembly', 5000],
  ['Export assembly', 5000],
];

const STL_STAGE_SPECS = [
  ['Auto-orient parts', 5000],
  ['Auto-stack parts', 5000],
  ['Auto-scale parts', 5000],
  ['Autodrop', 5000],
  ['Export parts', 5000],
  ['Render assembly', 5000],
  ['Export assembly', 5000],
];

function ensureArtifactDir() {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
}

async function switchToRepoRoot(page) {
  await page.getByRole('button', { name: /change dir/i }).click();
  await expect(page.locator('#dir-overlay')).toBeVisible();
  await page.getByRole('button', { name: /root/i }).click();
  await expect(page.locator('#dir-current')).toContainText('.');
  await page.getByRole('button', { name: /use this directory/i }).click();
  await expect(page.locator('#dir-overlay')).toBeHidden();
}

async function openModelDirectory(page) {
  await page.getByRole('button', { name: /change dir/i }).click();
  await expect(page.locator('#dir-overlay')).toBeVisible();
  await page.getByRole('button', { name: /root/i }).click();
  await page.locator('.dir-entry', { hasText: 'test_models' }).dblclick();
  await page.locator('.dir-entry', { hasText: 'conic_capsule_topopt_8' }).click();
  await page.getByRole('button', { name: /use this directory/i }).click();
  await expect(page.locator('#dir-overlay')).toBeHidden();
}

async function loadParts(page, fileNames) {
  await page.request.delete('/api/debug-log');
  for (const fileName of fileNames) {
    await page.locator('#file-list .file-row', { hasText: fileName }).click();
  }
  await page.getByRole('button', { name: /load selected/i }).click();
  await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout: 20000 });
  await expect(page.locator('#part-count')).toContainText(`${fileNames.length} part`);
  await expect(page.locator('#status')).toContainText(/loaded|scene ready/i);
  await assertNoHangSignals(page);
}

async function assertNoHangSignals(page) {
  const debugLog = await page.request.get('/api/debug-log');
  const payload = await debugLog.json();
  const text = (payload.entries || []).map((entry) => `${entry.kind}: ${entry.message}`).join('\n');
  expect(text).not.toContain('stall-warning');
  expect(text).not.toContain('scene-error');
  expect(text).not.toContain('load-error');
  expect(text).not.toContain('api-error');
}

async function runStageMeasured(page, buttonName, thresholdMs) {
  await page.request.delete('/api/debug-log');
  const button = page.getByRole('button', { name: new RegExp(buttonName, 'i') });
  await expect(button).toBeVisible();
  const started = Date.now();
  await button.click();
  await expect(button).toBeEnabled({ timeout: thresholdMs + 5000 });
  await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout: thresholdMs + 5000 });
  await expect(page.locator('.toast-error')).toHaveCount(0);
  await expect(page.locator('#status')).toContainText(/scene ready|complete|exported|rendered|done/i, {
    timeout: thresholdMs + 5000,
  });
  const elapsedMs = Date.now() - started;
  expect(elapsedMs, `${buttonName} exceeded ${thresholdMs}ms`).toBeLessThanOrEqual(thresholdMs);
  await assertNoHangSignals(page);
  return elapsedMs;
}

async function runStageSequence(page, specs) {
  const timings = {};
  for (const [name, thresholdMs] of specs) {
    timings[name] = await runStageMeasured(page, name, thresholdMs);
  }
  return timings;
}

async function loadAndShot(page, loaderFn, files, screenshotPath) {
  await loaderFn(page);
  await loadParts(page, files);
  await page.screenshot({ path: screenshotPath, fullPage: true });
}

test.describe('Pipeline Performance', () => {
  test('keeps all pipeline stages under 5s on STEP and STL pairs', async ({ page }) => {
    ensureArtifactDir();
    const allTimings = {};

    await page.goto('/');
    await switchToRepoRoot(page);
    await loadParts(page, ['outer_1.STEP', 'outer_2.STEP']);
    allTimings['outer_1.STEP + outer_2.STEP'] = await runStageSequence(page, STEP_STAGE_SPECS);

    await page.reload();
    await switchToRepoRoot(page);
    await loadParts(page, ['out.stl', 'out2.stl']);
    allTimings['out.stl + out2.stl'] = await runStageSequence(page, STL_STAGE_SPECS);

    fs.writeFileSync(TIMING_PATH, JSON.stringify(allTimings, null, 2));
  });

  test('survives three sequential load-and-pipeline cycles without hanging', async ({ page }) => {
    ensureArtifactDir();

    await page.goto('/');
    await switchToRepoRoot(page);
    await loadParts(page, ['outer_1.STEP', 'outer_2.STEP']);
    await runStageSequence(page, STEP_STAGE_SPECS);

    await page.reload();
    await loadAndShot(page, switchToRepoRoot, ['out.stl', 'out2.stl'], SECOND_SCREENSHOT);
    await runStageSequence(page, STL_STAGE_SPECS);

    await page.reload();
    await loadAndShot(page, openModelDirectory, ['outer_1.step', 'mid_1.step', 'inner_1.step'], THIRD_SCREENSHOT);
    await runStageSequence(page, [
      ['Auto-orient parts', 5000],
      ['Auto-stack parts', 5000],
      ['Auto-scale parts', 5000],
      ['Autodrop', 5000],
      ['Cut inner from mid', 5000],
      ['Export parts', 5000],
      ['Render assembly', 5000],
      ['Export assembly', 5000],
    ]);
  });
});
