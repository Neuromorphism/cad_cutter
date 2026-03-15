const { test, expect } = require('@playwright/test');

async function switchToRepoRoot(page) {
  await page.getByRole('button', { name: /change dir/i }).click();
  await expect(page.locator('#dir-overlay')).toBeVisible();
  await page.getByRole('button', { name: /root/i }).click();
  await page.getByRole('button', { name: /use this directory/i }).click();
  await expect(page.locator('#dir-overlay')).toBeHidden();
  await expect(page.locator('#parts-dir-label')).toContainText('/home/me/gits/cad_cutter');
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

async function clearSelections(page) {
  const boxes = page.locator('#file-list input[type="checkbox"]');
  const count = await boxes.count();
  for (let i = 0; i < count; i += 1) {
    const box = boxes.nth(i);
    if (await box.isChecked()) await box.uncheck();
  }
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
}

async function assertNoStageErrors(page) {
  const debugLog = await page.request.get('/api/debug-log');
  const payload = await debugLog.json();
  const text = (payload.entries || []).map((entry) => `${entry.kind}: ${entry.message}`).join('\n');
  expect(text).not.toContain('stall-warning');
  expect(text).not.toContain('scene-error');
  expect(text).not.toContain('load-error');
  expect(text).not.toContain('api-error');
}

async function runStage(page, buttonName, timeout = 30000) {
  const button = page.getByRole('button', { name: new RegExp(buttonName, 'i') });
  await page.request.delete('/api/debug-log');
  await button.click();
  await expect(button).toBeEnabled({ timeout });
  await expect(page.locator('#viewport-loader')).toHaveCount(0, { timeout });
  await expect(page.locator('.toast-error')).toHaveCount(0);
  await expect(page.locator('#status')).toContainText(/scene ready|complete|exported|rendered|done/i, { timeout });
  await assertNoStageErrors(page);
}

async function currentScene(page) {
  const response = await page.request.get('/api/scene');
  expect(response.ok()).toBeTruthy();
  return response.json();
}

test.describe('Repo Root Pipeline', () => {
  test('auto-stack on outer_1.STEP and outer_2.STEP keeps the stack ordered and does not fail', async ({ page }) => {
    await page.goto('/');
    await switchToRepoRoot(page);
    await loadParts(page, ['outer_1.STEP', 'outer_2.STEP']);
    await runStage(page, 'Auto-stack parts');

    const scene = await currentScene(page);
    const offsets = Object.fromEntries((scene.combined || []).map((entry) => [entry.name, entry.offset]));
    expect(offsets.outer_2[2]).toBeGreaterThan(offsets.outer_1[2]);
  });

  test('repo-root STEP pipeline stages do not fail with the same stage process', async ({ page }) => {
    const stages = [
      'Auto-orient parts',
      'Auto-stack parts',
      'Auto-scale parts',
      'Autodrop',
      'Export parts',
      'Render assembly',
      'Export assembly',
    ];

    for (const stage of stages) {
      await page.goto('/');
      await switchToRepoRoot(page);
      await loadParts(page, ['outer_1.STEP', 'outer_2.STEP']);
      await runStage(page, stage, 45000);
      const scene = await currentScene(page);
      expect(scene.parts.length).toBe(2);
      expect(scene.combined.length).toBe(2);
    }
  });

  test('cut stage uses the same no-fail process on a compatible test-model trio', async ({ page }) => {
    await page.goto('/');
    await openModelDirectory(page);
    await loadParts(page, ['outer_1.step', 'mid_1.step', 'inner_1.step']);
    await runStage(page, 'Cut inner from mid', 45000);
    const scene = await currentScene(page);
    expect(scene.parts.length).toBe(3);
    expect(scene.combined.length).toBe(3);
  });
});
