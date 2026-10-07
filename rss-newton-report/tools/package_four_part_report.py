from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, unquote
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import zipfile

ROOT=Path('/home/eheiden/.codex/visualizations/2026/10/07/01a113ce-dd5d-7a83-bb8a-8f0ea97a273b/rss-newton-report')
WORK=Path('/tmp/rss-newton-study')
soup=BeautifulSoup((ROOT/'index.html').read_text(),'html.parser')
ids=[e['id'] for e in soup.select('[id]')]
assert len(ids)==len(set(ids)), 'Duplicate IDs'
assert [s['id'] for s in soup.select('main>section')]==['overview','ecosystem','integration','recommendations']
assert 'The projects are on different integration tracks.' not in soup.get_text(' ',strip=True)
assert len(soup.select('#feature-table tbody tr'))==17
assert len(soup.select('video'))==4
assert len(list((ROOT/'examples').glob('*.py')))==6
assert len(soup.find_all(string=lambda s: s and 'PR #3850' in s))==1

qa_path=ROOT/'evidence/four-part-static-qa.json'
qa_path.write_text('{}\n') # The report links to this check's own result.
broken=[]
local_links=set()
for e in soup.select('[href], [src], [poster]'):
    for attr in ('href','src','poster'):
        value=e.get(attr)
        if not value: continue
        url=urlsplit(value)
        if url.scheme: continue
        local_links.add(value)
        if url.path:
            if not (ROOT/unquote(url.path)).exists(): broken.append(value)
        elif url.fragment and url.fragment not in ids:
            broken.append(value)
assert not broken, broken

repos={'KeplerC/RoboSimStudio':'RoboSimStudio','newton-physics/newton':'newton-pr3850','isaac-sim/IsaacLab':'IsaacLab','isaac-sim/IsaacSim':'IsaacSim','isaac-sim/IsaacLab-Arena':'IsaacLab-Arena','NVIDIA-Omniverse/ovnewton':'ovnewton','NVIDIA-Omniverse/ovnewton-internal':'ovnewton-internal'}
source_paths=[]
for url in sorted(set(e['href'] for e in soup.select('a[href^="https://github.com/"]'))):
    parts=urlsplit(url).path.strip('/').split('/')
    if len(parts)<5: continue
    repo='/'.join(parts[:2]);kind,sha=parts[2:4];path='/'.join(parts[4:])
    if repo not in repos: continue
    result=subprocess.run(['git','cat-file','-e',f'{sha}:{path}'],cwd=WORK/repos[repo],capture_output=True)
    assert result.returncode==0, (url,result.stderr.decode())
    fragment=urlsplit(url).fragment
    if fragment.startswith('L') and fragment[1:].isdigit():
        contents=subprocess.check_output(['git','show',f'{sha}:{path}'],cwd=WORK/repos[repo])
        assert int(fragment[1:]) <= len(contents.splitlines()),url
    source_paths.append(url)

baseline=json.loads((WORK/'report-before-four-part-manifest.json').read_text())
preserved=[]
for path, entry in baseline.items():
    if path in ('index.html','REPRODUCE.txt','manifest.json'): continue
    data=(ROOT/path).read_bytes()
    assert hashlib.sha256(data).hexdigest()==entry['sha256'],f'Original evidence changed: {path}'
    preserved.append(path)
previous=BeautifulSoup((WORK/'report-before-four-part.html').read_text(),'html.parser')
before=[row.find('td').get_text(' ',strip=True) for row in previous.select('#feature-table tbody tr')]
after=[row.find('td').get_text(' ',strip=True) for row in soup.select('#feature-table tbody tr')]
assert before==after
browser=json.loads((ROOT/'evidence/four-part-browser-qa.json').read_text())
assert browser['status']=='passed'
qa={'checked_at':datetime.now(timezone.utc).isoformat(),'status':'passed','sections':[s['id'] for s in soup.select('main>section')],'feature_rows':len(after),'video_count':4,'example_count':6,'old_section_absent':True,'one_brief_pr_baseline_note':True,'local_links_checked':len(local_links),'broken_local_links':broken,'pinned_source_paths_checked':len(source_paths),'pinned_source_paths':source_paths,'original_evidence_files_preserved_by_sha256':len(preserved),'preserved_files':preserved,'browser_check':'four-part-browser-qa.json','original_simulation_suites_rerun':False,'camera_demo_and_gui_probe_executed':True}
qa_path.write_text(json.dumps(qa,indent=2)+'\n')

manifest={}
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or path.name=='manifest.json': continue
    data=path.read_bytes()
    manifest[str(path.relative_to(ROOT))]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
bundle=ROOT.with_suffix('.zip')
temporary=bundle.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(ROOT.rglob('*')):
        if path.is_file():z.write(path,arcname=ROOT.name+'/'+str(path.relative_to(ROOT)))
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None
    assert z.read(ROOT.name+'/index.html')==(ROOT/'index.html').read_bytes()
    packed=json.loads(z.read(ROOT.name+'/manifest.json'))
    assert packed==manifest
    for path,entry in packed.items():
        data=z.read(ROOT.name+'/'+path)
        assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
    assert len(z.namelist())==len(manifest)+1
temporary.replace(bundle)
print(json.dumps({'status':'passed','preserved_original_files':len(preserved),'local_links_checked':len(local_links),'verified_pinned_sources':len(source_paths),'bundle_files':len(manifest)+1,'bundle_bytes':bundle.stat().st_size,'html_bytes':(ROOT/'index.html').stat().st_size},indent=2))
