const { test, expect } = require('@playwright/test');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ARTIFACT_DIR = path.resolve('.webui_cache', 'playwright_artifacts');
const ROOT_OUTER_STEP_SCREENSHOT = path.join(ARTIFACT_DIR, 'root-outer-step-visible.png');

function ensureArtifactDirs() {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
}

async function expectViewportToShowModel(page, screenshotPath) {
  ensureArtifactDirs();
  await page.locator('#main-canvas').screenshot({ path: screenshotPath });
  const metrics = JSON.parse(execFileSync('.venv/bin/python', ['-c', `
import json, math, sys
from PIL import Image

img = Image.open(sys.argv[1]).convert("RGB")
w, h = img.size
bg = img.getpixel((min(8, w - 1), min(8, h - 1)))
changed = 0
total = 0
stride = 2
for y in range(0, h, stride):
    for x in range(0, w, stride):
        total += 1
        px = img.getpixel((x, y))
        dist = math.sqrt(sum((px[i] - bg[i]) ** 2 for i in range(3)))
        if dist > 18:
            changed += 1
print(json.dumps({"changed": changed, "total": total}))
  `, screenshotPath], { encoding: 'utf8' }).trim());
  expect(metrics.changed / metrics.total).toBeGreaterThan(0.005);
}

async function openModelDirectory(page) {
  await page.getByRole('button', { name: /change dir/i }).click();
  await expect(page.locator('#dir-overlay')).toBeVisible();

  await page.getByRole('button', { name: /root/i }).click();
  await expect(page.locator('#dir-current')).toContainText('.');

  await page.locator('.dir-entry', { hasText: 'test_models' }).dblclick();
  await expect(page.locator('#dir-current')).toContainText('test_models');

  await page.locator('.dir-entry', { hasText: 'conic_capsule_topopt_8' }).click();
  await page.getByRole('button', { name: /use this directory/i }).click();

  await expect(page.locator('#dir-overlay')).toBeHidden();
  await expect(page.locator('#parts-dir-label')).toContainText('test_models/conic_capsule_topopt_8');
}

async function switchToRepoRoot(page) {
  await page.getByRole('button', { name: /change dir/i }).click();
  await expect(page.locator('#dir-overlay')).toBeVisible();
  await page.getByRole('button', { name: /root/i }).click();
  await expect(page.locator('#dir-current')).toContainText('.');
  await page.getByRole('button', { name: /use this directory/i }).click();
  await expect(page.locator('#dir-overlay')).toBeHidden();
  await expect(page.locator('#parts-dir-label')).toContainText('/home/me/gits/cad_cutter');
}

async function loadParts(page, fileNames) {
  for (const fileName of fileNames) {
    await page.locator('#file-list .file-row', { hasText: fileName }).click();
  }
  await page.getByRole('button', { name: /load selected/i }).click();
  await expect(page.locator('#part-count')).toContainText(`${fileNames.length} part`);
  await expect(page.locator('#status')).toContainText(/loaded|scene ready/i);
}

async function runStage(page, buttonName, { timeout = 20000 } = {}) {
  await page.request.delete('/api/debug-log');
  const button = page.getByRole('button', { name: new RegExp(buttonName, 'i') });
  await expect(button).toBeVisible();
  await button.click();
  await expect(button).toBeEnabled({ timeout });
  await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout });
  await expect(page.locator('.toast-error')).toHaveCount(0);
  await expect(page.locator('#status')).toContainText(/scene ready|complete|exported|rendered|done/i, { timeout });

  const debugLog = await page.request.get('/api/debug-log');
  const payload = await debugLog.json();
  const messages = (payload.entries || []).map((entry) => `${entry.kind}: ${entry.message}`);
  const logText = messages.join('\n');
  expect(logText).not.toContain('stall-warning');
  expect(logText).not.toContain('scene-error');
  expect(logText).not.toContain('load-error');
  expect(logText).not.toContain('api-error');
}

test.describe('CAD Cutter web UI', () => {
  test('loads the shell and keeps header tooltip visible in viewport', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('heading', { name: /cad cutter/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /change dir/i })).toBeVisible();
    await expect(page.locator('#webui-version')).toBeVisible();
    await expect(page.locator('#webui-version')).not.toHaveText('');
    await expect(page.locator('#workflow-select')).toBeVisible();
    await expect(page.locator('#fine-orient-toggle')).toBeVisible();
    await expect(page.getByRole('button', { name: /autodrop/i })).toBeVisible();

    const button = page.getByRole('button', { name: /change dir/i });
    await button.hover();
    await page.waitForTimeout(150);

    const tooltip = page.locator('#ui-tooltip');
    await expect(tooltip).toBeVisible();
    await expect(tooltip).toContainText('Change parts directory');

    const box = await tooltip.boundingBox();
    expect(box).not.toBeNull();
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.y + box.height).toBeLessThanOrEqual(980);
  });

  test('changes parts directory through the overlay and lists test model files', async ({ page }) => {
    await page.goto('/');
    await openModelDirectory(page);

    await expect(page.locator('#file-list')).toContainText('outer_1.step');
    await expect(page.locator('#file-list')).toContainText('mid_6.step');
    expect(await page.locator('#file-list .file-row').count()).toBeGreaterThanOrEqual(20);
  });

  test('loads selected parts and renders thumbnail cards', async ({ page }) => {
    await page.goto('/');
    await openModelDirectory(page);

    await page.locator('#file-list .file-row', { hasText: 'outer_1.step' }).click();
    await page.locator('#file-list .file-row', { hasText: 'inner_1.step' }).click();
    await page.getByRole('button', { name: /load selected/i }).click();

    await expect(page.locator('#viewport-progress-toggle')).toContainText('Recent activity');
    await expect(page.locator('#part-count')).toContainText('2 parts');
    await expect(page.locator('#thumbnails .thumb')).toHaveCount(2);
    await expect(page.locator('#status')).toContainText(/scene ready|loaded/i);
    await page.locator('#debug-toggle').click();
    await expect(page.locator('#debug-log')).toContainText('api-start');
    await expect(page.locator('#debug-log')).toContainText('/api/load');
    await expect(page.locator('#debug-log')).toContainText('Rendered combined view');
    await expect(page.getByRole('button', { name: /copy debug log/i })).toBeVisible();
  });

  test('loads large STL parts without falling back to a second scene request', async ({ page }) => {
    await page.goto('/');
    await switchToRepoRoot(page);

    await page.locator('#file-list .file-row', { hasText: 'out.stl' }).click();
    await page.locator('#file-list .file-row', { hasText: 'out2.stl' }).click();
    await page.getByRole('button', { name: /load selected/i }).click();

    await expect(page.locator('#part-count')).toContainText('2 parts');
    await expect(page.locator('#status')).toContainText(/loaded|scene ready/i);

    await page.locator('#debug-toggle').click();
    await expect(page.locator('#debug-log')).toContainText('Using scene payload returned from load');
    await expect(page.locator('#debug-log')).not.toContainText('GET /api/scene');
  });

  test('loads outer STEP sections and shows visible canvases', async ({ page }) => {
    await page.goto('/');
    await switchToRepoRoot(page);

    await page.locator('#file-list .file-row', { hasText: 'outer_1.STEP' }).click();
    await page.locator('#file-list .file-row', { hasText: 'outer_2.STEP' }).click();
    await page.getByRole('button', { name: /load selected/i }).click();

    await expect(page.locator('#part-count')).toContainText('2 parts');
    await expect(page.locator('#status')).toContainText(/loaded|scene ready/i);
    await expect(page.locator('#thumbnails .thumb')).toHaveCount(2);
    await expect(page.locator('.toast-error')).toHaveCount(0);
    await expectViewportToShowModel(page, ROOT_OUTER_STEP_SCREENSHOT);
    await page.locator('#debug-toggle').click();
    await expect(page.locator('#debug-log')).toContainText('Using scene payload returned from load');
    await expect(page.locator('#debug-log')).not.toContainText('/api/mesh-payload?path=outer_1.STEP');
  });

  test('auto-orients outer STEP sections through the browser without hanging', async ({ page }) => {
    await page.goto('/');
    await switchToRepoRoot(page);
    await loadParts(page, ['outer_1.STEP', 'outer_2.STEP']);

    await runStage(page, 'Auto-orient parts');

    await page.locator('#debug-toggle').click();
    await expect(page.locator('#debug-log')).toContainText('Auto-orient complete using axis-aligned six-way heuristic');
    await expect(page.locator('#debug-log')).toContainText('Rendered combined view');
  });

  test('runs the remaining pipeline stages through the browser without hanging', async ({ page }) => {
    await page.goto('/');
    await openModelDirectory(page);
    await loadParts(page, ['outer_1.step', 'mid_1.step', 'inner_1.step']);

    await runStage(page, 'Auto-stack parts');
    await runStage(page, 'Auto-scale parts');
    await runStage(page, 'Autodrop', { timeout: 25000 });
    await runStage(page, 'Cut inner from mid', { timeout: 25000 });
    await runStage(page, 'Export parts', { timeout: 25000 });
    await runStage(page, 'Render assembly', { timeout: 25000 });
    await runStage(page, 'Export assembly', { timeout: 25000 });

    await page.locator('#debug-toggle').click();
    await expect(page.locator('#debug-log')).toContainText('Exported web_assembly.step');
  });
});
