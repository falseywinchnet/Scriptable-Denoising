const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--autoplay-policy=no-user-gesture-required']});
 try {
 const page=await browser.newPage({viewport:{width:1080,height:1600}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url())});
 await page.goto(process.env.DEMO_URL||'http://127.0.0.1:8771/');
 await page.waitForFunction(()=>document.querySelector('#play').textContent==='Play'&&spectra.length===10&&spectra[9]?.surface_panels.every(p=>p.bytes),{timeout:60000});
 await page.locator('#span').selectOption('2');await page.locator('#hz').selectOption('4000');
 await page.locator('#seek').evaluate(e=>{e.value='2.7';e.dispatchEvent(new Event('input'))});
 await page.waitForFunction(()=>document.querySelector('#visualState').textContent.includes('2.700'));
 await page.locator('.visuals').screenshot({path:'build/listening-demo-v9/browser-occupancy.png'});
 for(const field of ['surface_mean','surface_variance','surface_low','surface_occupancy','surface_reference','surface_floor']){
  await page.locator('#surfaceField').selectOption(field);
  await page.waitForFunction(()=>!visualDirty);
 }
 for(const label of ['Surface + excitation','Occupancy + excitation']){
  await page.locator('#choices button').getByText(label,{exact:true}).click();
  await page.waitForFunction(label=>document.querySelector('#visualState').textContent.startsWith(label),label);
  await page.locator('#hz').selectOption('8000');await page.waitForFunction(()=>!visualDirty);
  await page.locator('.visuals').screenshot({path:'build/listening-demo-v9/browser-'+(label.startsWith('Surface')?'surface-excitation':'occupancy-excitation')+'.png'});
 }
 await page.locator('#play').click();await page.waitForFunction(()=>document.querySelector('#seek').value>3.1);
 await page.locator('#choices button').getByText('Variance surface',{exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#visualState').textContent.startsWith('Variance surface'));
 await page.locator('#play').click();const stopped=await page.locator('#seek').inputValue();
 await page.locator('#choices button').getByText('Baseline Cleanup',{exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#surfaceExtras').hidden);
 await page.locator('#choices button').getByText('Occupancy contour',{exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#visualState').textContent.startsWith('Occupancy contour')&&!document.querySelector('#surfaceExtras').hidden);
 const snapshot=await page.evaluate(()=>({buttons:[...document.querySelectorAll('#choices button')].map(b=>b.textContent),playing,position:document.querySelector('#seek').value,panels:spectra.flatMap(s=>[...s.panels,...(s.surface_panels||[])].map(p=>({kind:p.kind,bytes:p.bytes.length,expected:p.bins*p.frames}))),harmonics:[8,9].map(i=>({name:variants[i][0],activeBlocks:spectra[i].statistics.filter(s=>s.harmonic?.[0]>0).length}))}));
 if(errors.length||snapshot.playing||snapshot.buttons.length!==10||snapshot.panels.some(p=>p.bytes!==p.expected)||snapshot.position!==stopped||snapshot.harmonics.some(h=>!h.activeBlocks))throw Error(JSON.stringify({errors,snapshot}));
 fs.writeFileSync('docs/evidence/occupancy-excitation-browser.json',JSON.stringify({errors,...snapshot},null,2));
 console.log(JSON.stringify({errors,buttons:snapshot.buttons,harmonics:snapshot.harmonics,checkedPanels:snapshot.panels.length,paused:true}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
