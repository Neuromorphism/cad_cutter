const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/playwright',
  timeout: 120000,
  expect: {
    timeout: 10000,
  },
  fullyParallel: false,
  retries: 0,
  use: {
    baseURL: 'http://127.0.0.1:12084',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    viewport: { width: 1440, height: 980 },
  },
  webServer: {
    command: '.venv/bin/python web_ui.py --no-reload --host 127.0.0.1 --port 12084',
    url: 'http://127.0.0.1:12084',
    reuseExistingServer: true,
    timeout: 120000,
  },
});
