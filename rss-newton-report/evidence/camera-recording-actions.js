async (page) => {
 const root='/home/eheiden/.codex/visualizations/2026/10/07/01a113ce-dd5d-7a83-bb8a-8f0ea97a273b/rss-newton-report/';
 await page.mouse.move(636,522); await page.mouse.down();
 for(let i=1;i<=24;i++){await page.mouse.move(636,522-i*5);await page.waitForTimeout(55);}
 await page.mouse.up(); await page.waitForTimeout(2200);
 await page.screenshot({path:root+'media/rss-camera-hd-both.png'});
 await page.getByRole('combobox',{name:'Camera',exact:true}).click();
 await page.getByRole('option',{name:'overhead',exact:true}).click();
 await page.getByRole('button',{name:'Look through',exact:true}).click();
 await page.waitForTimeout(800);
 await page.getByRole('button',{name:'Release all',exact:true}).click();
 await page.waitForTimeout(1400);
 await page.screenshot({path:root+'media/rss-camera-hd-released.png'});
}
