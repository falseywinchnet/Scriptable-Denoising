const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--autoplay-policy=no-user-gesture-required']});
 try {
  const page=await browser.newPage({viewport:{width:1080,height:1700}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url())});
  await page.goto(process.env.DEMO_URL||'http://127.0.0.1:8769/');
  await page.waitForFunction(()=>document.querySelector('#play').textContent==='Play'&&spectra.length===14&&spectra[13]?.panels.every(p=>p.bytes),{timeout:60000});
  await page.locator('#choices button').getByText('Population Cleanup',{exact:true}).click();
  await page.locator('#span').selectOption('2');await page.locator('#hz').selectOption('4000');
  await page.locator('#seek').evaluate(e=>{e.value='2.7';e.dispatchEvent(new Event('input'))});
  await page.waitForFunction(()=>document.querySelector('#visualState').textContent.includes('2.700')&&!visualDirty);
  await page.getByText('Floor and peak statistics at this block',{exact:true}).click();
  await page.locator('#maskStats').waitFor({state:'visible'});
  await page.locator('.visuals').screenshot({path:'build/listening-demo-v16/browser-population.png'});
  const before=await page.evaluate(()=>({extras:document.querySelector('#surfaceExtras').hidden,legend:!document.querySelector('#populationKey').hidden,stats:document.querySelector('#maskStats').innerText,panels:spectra[13].panels.map(p=>({kind:p.kind,bytes:p.bytes.length,expected:p.bins*p.frames}))}));
  if(!before.extras||!before.legend||!before.stats.includes('Statistical population:'))throw Error(JSON.stringify(before));
  await page.locator('#play').click();await page.waitForFunction(()=>+document.querySelector('#seek').value>3.1);
  await page.locator('#choices button').getByText('Registered Cleanup',{exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#visualState').textContent.startsWith('Registered Cleanup'));
  await page.locator('#choices button').getByText('Population Cleanup',{exact:true}).click();
  await page.locator('#play').click();const stopped=await page.locator('#seek').inputValue();
  await page.locator('#choices button').getByText('Occupancy + comfort',{exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('#surfaceExtras').hidden&&!visualDirty);
  const snapshot=await page.evaluate(()=>({buttons:[...document.querySelectorAll('#choices button')].map(b=>b.textContent),playing,position:document.querySelector('#seek').value,finite:buffers.every(b=>Array.from(b.getChannelData(0)).every(Number.isFinite))}));
  if(errors.length||snapshot.playing||snapshot.buttons.length!==14||snapshot.position!==stopped||!snapshot.finite||before.panels.some(p=>p.bytes!==p.expected))throw Error(JSON.stringify({errors,snapshot,before}));
  fs.writeFileSync('docs/evidence/population-browser.json',JSON.stringify({errors,...snapshot,...before},null,2));
  console.log(JSON.stringify({errors,buttons:snapshot.buttons,paused:true,panels:before.panels}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
