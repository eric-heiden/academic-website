async (page) => {
  const root = '/home/eheiden/.codex/visualizations/2026/10/07/01a113ce-dd5d-7a83-bb8a-8f0ea97a273b/rss-newton-report/evidence/';
  const errors = [];
  const failedRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('response', response => { if (response.status() >= 400) failedRequests.push({url:response.url(),status:response.status()}); });
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('https://omnistation.tail159996.ts.net/rss-newton-report/');
  await page.waitForFunction(() => [...document.querySelectorAll('video')].every(v=>Number.isFinite(v.duration)));
  await page.screenshot({path:root+'four-part-desktop.png'});
  await page.locator('#overview .figure').screenshot({path:root+'four-part-architecture.png'});
  await page.locator('#ecosystem .figure').screenshot({path:root+'four-part-ecosystem.png'});
  await page.locator('#feature-handles').screenshot({path:root+'four-part-handles.png'});
  await page.locator('#roadmap').screenshot({path:root+'four-part-roadmap.png'});
  await page.locator('#matrix>summary').click();
  const count = () => page.locator('#feature-table tbody tr:not([hidden])').count();
  const matrix = {all:await count()};
  await page.locator('#feature-filter').selectOption('upstream');
  matrix.upstream = await count();
  await page.locator('#feature-filter').selectOption('all');
  await page.locator('#feature-search').fill('zz-no-matching-capability');
  matrix.noMatch = await count();
  await page.locator('#feature-search').fill('cloth');
  matrix.cloth = await count();
  await page.locator('#feature-search').fill('');
  matrix.restored = await count();
  await page.locator('#matrix>summary').click();
  const chapters = [];
  const chapterTargets = await page.locator('[data-video="rss-video"]').evaluateAll(nodes=>nodes.map(e=>e.dataset.time));
  for (const target of chapterTargets) {
    await page.locator('[data-video="rss-video"][data-time="'+target+'"]').click();
    await page.waitForFunction(t => {const v=document.getElementById('rss-video');return !v.seeking&&v.currentTime>=Number(t)&&v.currentTime<Number(t)+3;},target);
    chapters.push(await page.locator('#rss-video').evaluate(v=>({time:v.currentTime,playing:!v.paused,ready:v.readyState})));
    await page.locator('#rss-video').evaluate(v=>v.pause());
  }
  const videos=[];
  for (const id of ['rss-video','agent-video','newton-video','camera-video']) {
    await page.locator('#'+id).evaluate(async v=>{v.muted=true;v.currentTime=2;await v.play();});
    await page.waitForFunction(id=>{const v=document.getElementById(id);return v.currentTime>2.1&&!v.paused;},id);
    videos.push(await page.locator('#'+id).evaluate(v=>({id:v.id,src:v.currentSrc,poster:v.poster,duration:v.duration,currentTime:v.currentTime,width:v.videoWidth,height:v.videoHeight,playing:!v.paused,error:v.error?.message||null})));
    await page.locator('#'+id).evaluate(v=>{v.currentTime=v.duration-0.3;});
    await page.waitForFunction(id=>document.getElementById(id).ended,id);
    await page.locator('#'+id).evaluate(v=>v.pause());
  }
  const navigation=[];
  for (const id of ['overview','ecosystem','integration','recommendations']) {
    await page.locator('.rail nav a[href="#'+id+'"]').click();
    await page.waitForFunction(id=>location.hash==='#'+id&&document.querySelector('.rail nav a.active')?.getAttribute('href')==='#'+id,id);
    navigation.push(id);
  }
  await page.setViewportSize({width:390,height:844});
  await page.goto('https://omnistation.tail159996.ts.net/rss-newton-report/');
  await page.evaluate(()=>{document.documentElement.style.scrollBehavior='auto';window.scrollTo(0,0);});
  await page.waitForFunction(()=>window.scrollY===0);
  await page.screenshot({path:root+'four-part-mobile.png'});
  const mobile={width:390,scrollWidth:await page.evaluate(()=>document.documentElement.scrollWidth)};
  await page.locator('#feature-handles').scrollIntoViewIfNeeded();
  await page.locator('#feature-handles').screenshot({path:root+'four-part-handles-mobile.png'});
  await page.locator('#roadmap .phase').first().scrollIntoViewIfNeeded();
  await page.screenshot({path:root+'four-part-roadmap-mobile.png'});
  await page.locator('#ecosystem .figure').scrollIntoViewIfNeeded();
  mobile.diagram=await page.locator('#ecosystem .diagram-scroll').evaluate(e=>({clientWidth:e.clientWidth,scrollWidth:e.scrollWidth}));
  await page.locator('#matrix>summary').click();
  mobile.matrix=await page.locator('#matrix .table-wrap').evaluate(e=>({clientWidth:e.clientWidth,scrollWidth:e.scrollWidth}));
  mobile.widthWithMatrixOpen=await page.evaluate(()=>document.documentElement.scrollWidth);
  await page.locator('#matrix>summary').click();
  for(const id of ['acceptance','evidence','sources']){
    await page.locator('#'+id+'>summary').click();
    if (!(await page.locator('#'+id).evaluate(e=>e.open))) throw new Error('Details did not open: '+id);
    await page.locator('#'+id+'>summary').click();
  }
  const images = await page.locator('img').evaluateAll(nodes=>nodes.map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0})));
  const sections = await page.locator('main>section').evaluateAll(nodes=>nodes.map(e=>e.id));
  if(matrix.all!==17||matrix.upstream!==3||matrix.noMatch!==0||matrix.restored!==17)throw new Error('Matrix assertion failed');
  if(mobile.scrollWidth!==390||mobile.widthWithMatrixOpen!==390)throw new Error('Mobile page overflow');
  if(errors.length||failedRequests.length)throw new Error(JSON.stringify({errors,failedRequests}));
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(()=>window.scrollTo(0,0));
  return {report:'four-part revision',checkedAt:new Date().toISOString(),sections,matrix,chapters,videos,navigation,mobile,images,consoleErrors:errors,failedRequests,status:'passed'};
}
