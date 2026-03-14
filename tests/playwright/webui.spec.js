const { test, expect } = require('@playwright/test');

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

test.describe('CAD Cutter web UI', () => {
  test('loads the shell and keeps header tooltip visible in viewport', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByRole('heading', { name: /cad cutter/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /change dir/i })).toBeVisible();
    await expect(page.locator('#webui-version')).toBeVisible();
    await expect(page.locator('#webui-version')).not.toHaveText('');
    await expect(page.locator('#workflow-select')).toBeVisible();
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
  });
});
