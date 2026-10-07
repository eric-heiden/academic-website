async (page) => {
 const root='/home/eheiden/.codex/visualizations/2026/10/07/01a113ce-dd5d-7a83-bb8a-8f0ea97a273b/rss-newton-report/evidence/';
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.setViewportSize({width:1440,height:1000});
 await page.goto('https://omnistation.tail159996.ts.net/rss-newton-report/');
 await page.locator('#api-comparison h3').first().scrollIntoViewIfNeeded();
 await page.screenshot({path:root+'engineering-api-desktop.png'});
 await page.locator('#api-comparison pre').screenshot({path:root+'engineering-code-desktop.png'});
 await page.locator('#gui-editing .gui-explainer').screenshot({path:root+'engineering-gui-desktop.png'});
 await page.locator('#application-structure .figure').screenshot({path:root+'engineering-application-diagram.png'});
 const highlighted=await page.locator('code').evaluateAll(xs=>({count:xs.length,plain:xs.filter(x=>!x.classList.contains('syntax-python')&&!x.classList.contains('revision')).map(x=>x.textContent),blocks:xs.filter(x=>x.parentElement.tagName==='PRE').map(x=>({text:x.textContent,tokens:x.querySelectorAll('span').length,colors:[...new Set([...x.querySelectorAll('span')].map(s=>getComputedStyle(s).color))]}))}));
 if(highlighted.plain.length)throw new Error('Unhighlighted code: '+JSON.stringify(highlighted.plain));
 const downloaded=await (await page.request.get('https://omnistation.tail159996.ts.net/rss-newton-report/examples/rss_cloth_cameras.py')).text();
 const expected='import robosimstudio as rss'+downloaded.split('import robosimstudio as rss')[1];
 if(highlighted.blocks.length!==1||highlighted.blocks[0].text!==expected||highlighted.blocks[0].colors.length<4)throw new Error('Snippet/highlighting mismatch');
 const video=await page.locator('#camera-video').evaluate(async v=>{v.muted=true;await v.play();return {duration:v.duration,width:v.videoWidth,height:v.videoHeight,playing:!v.paused};});
 await page.waitForTimeout(500);await page.locator('#camera-video').evaluate(v=>v.pause());
 const newImages=await page.locator('#gui-editing img,#feature-cameras img').evaluateAll(xs=>xs.map(x=>({src:x.getAttribute('src'),width:x.naturalWidth,height:x.naturalHeight,loaded:x.complete&&x.naturalWidth>0})));
 if(newImages.some(x=>!x.loaded))throw new Error('Image missing');
 await page.setViewportSize({width:390,height:844});
 await page.locator('#api-comparison pre').scrollIntoViewIfNeeded();await page.screenshot({path:root+'engineering-code-mobile.png'});
 await page.locator('#gui-editing').scrollIntoViewIfNeeded();await page.screenshot({path:root+'engineering-gui-mobile.png'});
 const width=await page.evaluate(()=>document.documentElement.scrollWidth);
 if(width!==390||errors.length)throw new Error(JSON.stringify({width,errors}));
 return {status:'passed',checkedAt:new Date().toISOString(),highlighted,video,newImages,mobileWidth:width,errors};
}
