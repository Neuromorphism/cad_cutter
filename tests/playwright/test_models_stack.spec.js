const { test, expect } = require('@playwright/test');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ARTIFACT_DIR = path.resolve('.webui_cache', 'playwright_artifacts');
const UPLOAD_DIR = path.resolve('.webui_cache', 'playwright_uploads');
const MODEL_DIR = path.resolve('test_models', 'conic_capsule_topopt_8');
const AUTOLOAD_SCREENSHOT = path.join(ARTIFACT_DIR, 'test-model-autoload.png');
const TWO_OUTER_UI_SCREENSHOT = path.join(ARTIFACT_DIR, 'test-model-two-outer-ui.png');
const TWO_OUTER_RENDER = path.join(ARTIFACT_DIR, 'test-model-two-outer-render.png');
const FULL_STACK_UI_SCREENSHOT = path.join(ARTIFACT_DIR, 'test-model-full-stack-ui.png');
const FULL_STACK_RENDER = path.join(ARTIFACT_DIR, 'test-model-full-stack-render.png');

function ensureArtifactDirs() {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
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

async function setPartsDir(page, dirPath) {
  const resp = await page.request.patch('/api/parts-dir', { data: { path: dirPath } });
  expect(resp.ok()).toBeTruthy();
  await page.reload();
}

async function clearSelections(page) {
  const boxes = page.locator('#file-list input[type="checkbox"]');
  const count = await boxes.count();
  for (let i = 0; i < count; i += 1) {
    const box = boxes.nth(i);
    if (await box.isChecked()) await box.uncheck();
  }
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

async function waitForRenderReady(page, timeout = 10000) {
  await expect
    .poll(async () => {
      const debugLog = await page.request.get('/api/debug-log');
      const payload = await debugLog.json();
      return (payload.entries || []).some((entry) => entry.kind === 'render-ready');
    }, { timeout })
    .toBeTruthy();
}

async function loadParts(page, fileNames) {
  await clearSelections(page);
  for (const fileName of fileNames) {
    await page.locator('#file-list .file-row', { hasText: fileName }).click();
  }
  await page.request.delete('/api/debug-log');
  await page.getByRole('button', { name: /load selected/i }).click();
  await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout: 30000 });
  await expect(page.locator('#part-count')).toContainText(`${fileNames.length} part`);
  await expect(page.locator('.toast-error')).toHaveCount(0);
  await expect(page.locator('#status')).toContainText(/loaded|scene ready/i);
  await waitForRenderReady(page, 30000);
  await assertNoHangSignals(page);
}

async function runStage(page, buttonName, timeout = 30000) {
  await page.request.delete('/api/debug-log');
  const button = page.getByRole('button', { name: new RegExp(buttonName, 'i') });
  await button.click();
  await expect(button).toBeEnabled({ timeout });
  await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout });
  await expect(page.locator('.toast-error')).toHaveCount(0);
  await expect(page.locator('#status')).toContainText(/scene ready|complete|exported|rendered|done/i, { timeout });
  await waitForRenderReady(page, timeout);
  await assertNoHangSignals(page);
}

async function currentScene(page) {
  const response = await page.request.get('/api/scene');
  expect(response.ok()).toBeTruthy();
  return response.json();
}

function bboxFromPart(part, offset = [0, 0, 0]) {
  const vertices = part.mesh?.vertices || [];
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 3) {
    const x = vertices[i] + offset[0];
    const y = vertices[i + 1] + offset[1];
    const z = vertices[i + 2] + offset[2];
    if (x < mins[0]) mins[0] = x;
    if (y < mins[1]) mins[1] = y;
    if (z < mins[2]) mins[2] = z;
    if (x > maxs[0]) maxs[0] = x;
    if (y > maxs[1]) maxs[1] = y;
    if (z > maxs[2]) maxs[2] = z;
  }
  return { mins, maxs };
}

function containsBBox(outer, inner, tol = 2) {
  return inner.mins[0] >= outer.mins[0] - tol
    && inner.mins[1] >= outer.mins[1] - tol
    && inner.mins[2] >= outer.mins[2] - tol
    && inner.maxs[0] <= outer.maxs[0] + tol
    && inner.maxs[1] <= outer.maxs[1] + tol
    && inner.maxs[2] <= outer.maxs[2] + tol;
}

function combinedOffsetMap(scene) {
  return Object.fromEntries((scene.combined || []).map((entry) => [entry.name, entry.offset || [0, 0, 0]]));
}

function partMap(scene) {
  return Object.fromEntries((scene.parts || []).map((part) => [part.name, part]));
}

function copyIfExists(sourcePath, targetPath) {
  if (fs.existsSync(sourcePath)) fs.copyFileSync(sourcePath, targetPath);
}

async function expectViewportToShowModel(page, screenshotPath) {
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
  const { changed, total } = metrics;
  expect(changed / total).toBeGreaterThan(0.005);
}

test.describe('Test Models Stack', () => {
  test('upload autoload completes without timing out', async ({ page }) => {
    ensureArtifactDirs();
    await page.goto('/');
    await setPartsDir(page, UPLOAD_DIR);
    await expect(page.locator('#parts-dir-label')).toContainText('.webui_cache/playwright_uploads');

    await page.request.delete('/api/debug-log');
    await page.locator('#browse-part-input').setInputFiles(path.join(MODEL_DIR, 'outer_1.step'));

    await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout: 30000 });
    await expect(page.locator('#part-count')).toContainText('1 part');
    await expect(page.locator('#status')).toContainText(/loaded|scene ready/i);
    await waitForRenderReady(page, 30000);
    await expect(page.locator('.toast-error')).toHaveCount(0);
    await assertNoHangSignals(page);
    await page.screenshot({ path: AUTOLOAD_SCREENSHOT, fullPage: true });
  });

  test('two outer test models autostack into a vertical assembly and render correctly', async ({ page }) => {
    ensureArtifactDirs();
    await page.goto('/');
    await openModelDirectory(page);
    await loadParts(page, ['outer_1.step', 'outer_2.step']);
    await runStage(page, 'Auto-stack parts');

    const scene = await currentScene(page);
    const offsets = combinedOffsetMap(scene);
    expect(offsets.outer_1[2]).toBeLessThan(offsets.outer_2[2]);
    expect(offsets.outer_1[2]).toBeGreaterThan(200);
    expect(offsets.outer_2[2]).toBeGreaterThan(700);

    await expectViewportToShowModel(page, TWO_OUTER_UI_SCREENSHOT);
    await runStage(page, 'Render assembly', 45000);
    copyIfExists(path.resolve('web_render.png'), TWO_OUTER_RENDER);
  });

  test('full test model assembly stacks into a cone with nested mids and inners', async ({ page }) => {
    ensureArtifactDirs();
    await page.goto('/');
    await openModelDirectory(page);

    const files = [
      ...Array.from({ length: 8 }, (_, i) => `outer_${i + 1}.step`),
      ...Array.from({ length: 6 }, (_, i) => `mid_${i + 1}.step`),
      ...Array.from({ length: 6 }, (_, i) => `inner_${i + 1}.step`),
    ];
    await loadParts(page, files);
    await runStage(page, 'Auto-stack parts');

    const scene = await currentScene(page);
    const offsets = combinedOffsetMap(scene);
    const parts = partMap(scene);

    for (let i = 1; i < 8; i += 1) {
      expect(offsets[`outer_${i}`][2]).toBeLessThan(offsets[`outer_${i + 1}`][2]);
    }

    for (let level = 1; level <= 6; level += 1) {
      const outerBBox = bboxFromPart(parts[`outer_${level}`], offsets[`outer_${level}`]);
      const midBBox = bboxFromPart(parts[`mid_${level}`], offsets[`mid_${level}`]);
      const innerBBox = bboxFromPart(parts[`inner_${level}`], offsets[`inner_${level}`]);
      expect(containsBBox(outerBBox, midBBox)).toBeTruthy();
      expect(containsBBox(midBBox, innerBBox)).toBeTruthy();
    }

    await expectViewportToShowModel(page, FULL_STACK_UI_SCREENSHOT);
    await runStage(page, 'Render assembly', 45000);
    copyIfExists(path.resolve('web_render.png'), FULL_STACK_RENDER);
  });
});
