async (page) => {
 const out='/home/eheiden/.codex/visualizations/2026/10/07/01a113ce-dd5d-7a83-bb8a-8f0ea97a273b/rss-newton-report/evidence/';
 const errors=[];const failed=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('response',r=>{if(r.status()>=400)failed.push({url:r.url(),status:r.status()});});
 await page.setViewportSize({width:1440,height:1000});
 await page.goto('https://omnistation.tail159996.ts.net/rss-newton-report/#component-integration');
 await page.evaluate(()=>{document.documentElement.style.scrollBehavior='auto';document.getElementById('component-integration').scrollIntoView({block:'start'});});
 const rows=await page.locator('.component-table tbody tr').count();
 if(rows!==8)throw new Error('Expected eight component rows');
 await page.screenshot({path:out+'component-table-desktop.png'});
 const headings=await page.locator('.component-table th').allTextContents();
 const previews=[];
 for(let i=0;i<3;i++){
   const link=page.locator('.component-shot').nth(i);
   await link.click();
   await page.waitForFunction(()=>{const d=document.getElementById('component-image-dialog'),i=document.getElementById('component-image');return d.open&&i.complete&&i.naturalWidth>0;});
   const info=await page.locator('#component-image').evaluate(i=>({src:i.getAttribute('src'),width:i.naturalWidth,height:i.naturalHeight}));
   const caption=await page.locator('#component-image-caption').textContent();
   if(!caption||!info.width)throw new Error('Missing expanded image/caption');
   if(i===1)await page.screenshot({path:out+'component-table-expanded.png'});
   if(i===0)await page.keyboard.press('Escape');
   else if(i===1)await page.getByRole('button',{name:'Close enlarged screenshot'}).click();
   else await page.mouse.click(3,3);
   if(await page.locator('#component-image-dialog').evaluate(d=>d.open))throw new Error('Dialog failed to close');
   if(!(await link.evaluate(e=>document.activeElement===e)))throw new Error('Focus was not restored');
   previews.push({...info,caption});
 }
 await page.setViewportSize({width:390,height:844});
 await page.locator('#component-integration').scrollIntoViewIfNeeded();
 await page.screenshot({path:out+'component-table-mobile.png'});
 const layout=await page.evaluate(()=>({page:document.documentElement.scrollWidth,table:document.querySelector('.component-table-wrap').scrollWidth,viewport:document.querySelector('.component-table-wrap').clientWidth}));
 if(layout.page!==390||layout.table<=layout.viewport)throw new Error(JSON.stringify(layout));
 await page.locator('.component-shot').first().click();
 await page.waitForFunction(()=>{const i=document.getElementById('component-image');return i.complete&&i.naturalWidth>0;});
 await page.screenshot({path:out+'component-table-expanded-mobile.png'});
 const modal=await page.locator('#component-image-dialog').boundingBox();
 if(modal.x<0||modal.x+modal.width>390)throw new Error('Dialog overflows mobile');
 await page.keyboard.press('Escape');
 const codes=await page.locator('code').evaluateAll(xs=>({total:xs.length,unhighlighted:xs.filter(e=>!e.classList.contains('syntax-python')&&!e.classList.contains('revision')).map(e=>e.textContent)}));
 if(codes.unhighlighted.length||errors.length||failed.length)throw new Error(JSON.stringify({codes,errors,failed}));
 return {status:'passed',checkedAt:new Date().toISOString(),rows,headings,previews,desktopWidth:1440,mobile:layout,modal,codes,errors,failed};
}
