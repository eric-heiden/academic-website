const { chromium } = require('/home/horde/.cache/npm/_npx/e41f203b7505f1fb/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');

(async function () {
  const root = path.resolve(__dirname, '..');
  const output = { checked: new Date().toISOString(), viewports: [], interactions: {}, errors: [] };
  const browser = await chromium.launch({ headless: true, executablePath: '/home/horde/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome' });
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on('pageerror', error => output.errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') output.errors.push(message.text()); });
  page.on('response', response => { if (response.status() >= 400) output.errors.push(`${response.status()} ${response.url()}`); });
  const address = 'http://127.0.0.1:8921/newton-live-mcp/';
  await page.goto(address, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.querySelectorAll('img').forEach(image => image.loading = 'eager'));
  output.interactions.first_visit = await page.evaluate(() => ({ theme: document.documentElement.dataset.theme, background: getComputedStyle(document.body).backgroundColor }));
  for (const width of [1024, 736, 360]) {
    await page.setViewportSize({ width, height: 900 });
    for (const theme of ['light', 'dark']) {
      if (await page.evaluate(() => document.documentElement.dataset.theme) !== theme) await page.locator('#theme-toggle').click();
      await page.waitForTimeout(350);
      await page.evaluate(async () => {
        for (let y = 0; y < document.body.scrollHeight; y += 700) {
          window.scrollTo(0, y);
          await new Promise(resolve => setTimeout(resolve, 25));
        }
      });
      await page.waitForFunction(() => [...document.images].every(image => image.complete && image.naturalWidth > 0));
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(150);
      const metrics = await page.evaluate(() => ({
        width: innerWidth,
        theme: document.documentElement.dataset.theme,
        page_width: document.documentElement.scrollWidth,
        background: getComputedStyle(document.body).backgroundColor,
        title_px: getComputedStyle(document.querySelector('h1')).fontSize,
        diagram: !!document.querySelector('.mermaid-shell svg'),
        plot: !!document.querySelector('#systems-plot .main-svg') && !!document.querySelector('#confirmation-ratios .main-svg') && !!document.querySelector('#calibration-convergence .main-svg'),
        images: [...document.images].map(image => ({ src: image.getAttribute('src'), loaded: image.complete && image.naturalWidth > 0 })),
        outside: [...document.querySelectorAll('main > *, .report-section, .figure-caption, .channel-control, video')].filter(element => element.getBoundingClientRect().right > innerWidth + 1).map(element => element.id || element.tagName)
      }));
      output.viewports.push(metrics);
      await page.screenshot({ path: path.join(__dirname, `${width}-${theme}.png`), fullPage: true });
    }
  }
  await page.reload({ waitUntil: 'networkidle' });
  output.interactions.dark_persists = await page.evaluate(() => document.documentElement.dataset.theme === 'dark');
  output.interactions.ratios = [];
  for (const include of [true, false]) {
    await page.locator('#ratio-uncached').setChecked(include);
    await page.waitForTimeout(100);
    output.interactions.ratios.push(await page.locator('#confirmation-ratios').evaluate(node => ({ series: node.data.length, pairs: node.data[0].x.length, values: node.data.map(item => item.x) })));
  }
  output.interactions.convergence = [];
  for (const instance of ['0', '1', '2']) {
    await page.selectOption('#calibration-instance', instance);
    await page.waitForTimeout(100);
    output.interactions.convergence.push(await page.locator('#calibration-convergence').evaluate(node => node.data.map(item => ({ samples: item.x.length, errors: item.y }))));
  }
  await page.selectOption('#calibration-instance', '0');
  output.interactions.asset_observations = [];
  for (const details of await page.locator('.asset-observations').all()) {
    await details.locator('summary').click();
    await page.waitForTimeout(100);
    output.interactions.asset_observations.push(await details.evaluate(node => ({ open: node.open, images: [...node.querySelectorAll('img')].every(image => image.complete && image.naturalWidth > 0) })));
  }
  output.interactions.systems_plot = [];
  for (const scene of ['panda', 'allegro', 'hug']) {
    await page.selectOption('#systems-task', scene);
    for (const include of [true, false]) {
      await page.locator('#systems-startup').setChecked(include);
      await page.waitForTimeout(100);
      output.interactions.systems_plot.push(await page.evaluate(() => ({ scene: document.querySelector('#systems-task').value, startup_included: document.querySelector('#systems-startup').checked, live: document.querySelector('#systems-plot').data[0].y, restart: document.querySelector('#systems-plot').data[1].y, accessible_label: document.querySelector('#systems-plot').getAttribute('aria-label') })));
    }
  }
  await page.locator('#systems-startup').setChecked(true);
  await page.locator('.development-detail summary').click();
  output.interactions.development_details = await page.locator('.development-detail').evaluate(node => node.open);
  output.interactions.expanded_overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  await page.locator('#theme-toggle').focus();
  await page.keyboard.press('Enter');
  output.interactions.keyboard_theme = await page.evaluate(() => document.documentElement.dataset.theme === 'light');
  const options = await page.locator('#channel-select option').evaluateAll(options => options.map(option => option.value));
  output.interactions.channels = [];
  for (const value of options) {
    await page.selectOption('#channel-select', value);
    await page.waitForFunction(() => document.querySelector('#channel-image').complete && document.querySelector('#channel-image').naturalWidth > 0);
    output.interactions.channels.push(await page.evaluate(() => ({ value: document.querySelector('#channel-select').value, image: document.querySelector('#channel-image').getAttribute('src'), caption: document.querySelector('#channel-detail').textContent })));
  }
  await page.locator('[data-copy-target="embed-code"]').click();
  await page.waitForFunction(() => document.querySelector('[data-copy-target="embed-code"]').textContent === 'Copied');
  output.interactions.copy_control = await page.locator('[data-copy-target="embed-code"]').textContent();
  output.interactions.video = [];
  for (const videoNode of await page.locator('video').all()) output.interactions.video.push(await videoNode.evaluate(async video => {
    await video.play();
    await new Promise(resolve => setTimeout(resolve, 350));
    video.pause();
    return { ready_state: video.readyState, duration: video.duration, time: video.currentTime, error: video.error?.message || null };
  }));
  await page.locator('.contents a[href="#method"]').click();
  output.interactions.navigation = page.url().endsWith('#method');
  await page.goto('http://127.0.0.1:8921/', { waitUntil: 'networkidle' });
  output.interactions.overview_title = await page.locator('a[href="newton-live-mcp/"] strong').textContent();
  await page.locator('a[href="newton-live-mcp/"]').click();
  output.interactions.overview_link = page.url() === address;
  output.interactions.local_links = [];
  for (const link of await page.locator('main a[href]').evaluateAll(links => links.map(link => link.getAttribute('href')))) {
    if (link.startsWith('http') || link.startsWith('#')) continue;
    output.interactions.local_links.push({ link, exists: fs.existsSync(path.resolve(root, link)) });
  }
  fs.writeFileSync(path.join(__dirname, 'browser-checks.json'), JSON.stringify(output, null, 2) + '\n');
  const failed = output.errors.length || output.viewports.some(value => value.page_width > value.width || !value.diagram || !value.plot || value.outside.length || value.images.some(image => !image.loaded)) || output.interactions.expanded_overflow || output.interactions.local_links.some(link => !link.exists) || output.interactions.video.some(video => video.error || video.ready_state < 2 || video.time <= 0) || output.interactions.asset_observations.some(item => !item.open || !item.images) || output.interactions.ratios.some(item => item.pairs !== 9) || output.interactions.convergence.some(pair => pair.some(series => series.samples !== 9));
  console.log(JSON.stringify({ viewports: output.viewports.map(({ width, theme, page_width, background, diagram }) => ({ width, theme, page_width, background, diagram })), interactions: output.interactions, errors: output.errors, passed: !failed }, null, 2));
  await browser.close();
  process.exitCode = failed ? 1 : 0;
})();
