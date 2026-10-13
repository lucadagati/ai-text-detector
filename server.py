"""AI-text detector service.

  GET  /              web page (paste text or upload .tex/.md/.txt)
  POST /api/detect    {"text": "...", "min_words": 40, "latex": null}  -> verdict, paragraphs, sentences, traits
  GET  /api/health    model info and whether the models are loaded
Models load on the first request and are released after AIDET_IDLE_MIN minutes of inactivity
(default 15), because the GPU is shared.
"""
import os, threading, time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from aidet.ensemble import MODEL, Detector

IDLE = float(os.environ.get("AIDET_IDLE_MIN", "15")) * 60
det = Detector()
last = [time.time()]
app = FastAPI(title="ai-detector")


class Req(BaseModel):
    text: str
    min_words: int = 40
    latex: bool | None = None


@app.post("/api/detect")
def detect(r: Req):
    last[0] = time.time()
    res = det.analyze(r.text, r.min_words, r.latex)
    last[0] = time.time()
    return res


@app.get("/api/health")
def health():
    return {"loaded": det.loaded, "idle_unload_min": IDLE / 60, "type": MODEL.get("type", "single"),
            "models": [m["model"] for m in MODEL["members"]] if MODEL.get("type") == "or" else [MODEL.get("ft_model")]}


def _reaper():
    while True:
        time.sleep(30)
        if det.loaded and time.time() - last[0] > IDLE:
            det.unload()


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Text Detector</title><style>
:root{--bg:#f7f6f3;--card:#fff;--ink:#1f1f1d;--mut:#6b6a66;--line:#e2e0da;--ai:#c62e2e;--unc:#d98a1c;--hu:#1b8f62;--acc:#2a78d6;--hl:rgba(198,46,46,.16);--hl2:rgba(217,138,28,.16)}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--mut:#a3a19b;--line:#33322f;--ai:#ef6b6b;--unc:#f0b04a;--hu:#4cc794;--acc:#6aa8f0;--hl:rgba(239,107,107,.22);--hl2:rgba(240,176,74,.2)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 18px}
textarea{width:100%;min-height:200px;padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font:13px/1.45 ui-monospace,monospace}
.row{display:flex;gap:10px;align-items:center;margin:10px 0 20px;flex-wrap:wrap}button,label.btn{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:9px 16px;font-weight:600;cursor:pointer}
label.btn{background:transparent;color:var(--acc);border:1px solid var(--acc)}input[type=file]{display:none}#st{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
.verdict{font-size:20px;font-weight:700}.bar{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:10px 0 6px;background:var(--line)}
.leg{display:flex;gap:16px;color:var(--mut);font-size:13px;flex-wrap:wrap}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
h2{font-size:15px;margin:0 0 8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
.p{border-left:4px solid var(--line);padding:8px 12px;margin:10px 0;background:var(--card);border-radius:0 8px 8px 0}
.p.ai{border-color:var(--ai)}.p.uncertain{border-color:var(--unc)}.p.human{border-color:var(--hu)}
.meta{font-size:12px;color:var(--mut);margin-bottom:4px}.meta b{text-transform:uppercase}
.s.hi{background:var(--hl);border-radius:3px}.s.mid{background:var(--hl2);border-radius:3px}
.note{font-size:12.5px;color:var(--mut)}.tabs{display:flex;gap:8px;margin:6px 0}</style></head><body><main>
<h1>AI Text Detector</h1><p class="sub">Domain-tuned classifier plus supervised detectors and a sentence-structure profile. Accepts plain text, Markdown or LaTeX (<code>changes</code> markup is resolved to the accepted text).</p>
<textarea id="t" placeholder="Paste text or LaTeX here..."></textarea>
<div class="row"><button id="go">Analyze</button><label class="btn">Upload file<input type="file" id="f" accept=".tex,.md,.txt"></label><span id="st"></span></div>
<div id="out"></div>
<p class="note">Scores are probabilistic. A paragraph is labelled AI above the threshold that flags 1% of human paragraphs in calibration; "uncertain" lies between the 1% and 5% thresholds. Highlighted sentences sit in 3-sentence windows that score above the same 1% level. No result is proof of authorship.</p>
</main><script>
const $=s=>document.querySelector(s);
$('#f').onchange=e=>{const r=new FileReader();r.onload=()=>{$('#t').value=r.result};r.readAsText(e.target.files[0])};
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let D=null,order='doc';
function render(){const d=D,s=d.share_words,pc=x=>(100*x).toFixed(0)+'%';
 let h=`<div class="card"><div class="verdict">${d.verdict}</div><div class="bar"><div style="width:${pc(s.ai)};background:var(--ai)"></div><div style="width:${pc(s.uncertain)};background:var(--unc)"></div><div style="width:${pc(s.human)};background:var(--hu)"></div></div>
 <div class="leg"><span><i class="dot" style="background:var(--ai)"></i>AI ${pc(s.ai)}</span><span><i class="dot" style="background:var(--unc)"></i>uncertain ${pc(s.uncertain)}</span><span><i class="dot" style="background:var(--hu)"></i>human ${pc(s.human)}</span><span>${d.n_paragraphs} paragraphs, share of words${d.n_skipped?` · ${d.n_skipped} math- or code-heavy paragraphs not assessed`:''}</span></div></div>`;
 if(d.structure_traits.length){h+=`<div class="card"><h2>Sentence structure, farthest from human academic writing</h2><table><tr><th>trait</th><th class="n">this text</th><th class="n">human papers</th><th class="n">z</th></tr>`;
  for(const t of d.structure_traits)h+=`<tr><td>${esc(t.trait)}</td><td class="n">${t.value}</td><td class="n">${t.human_mean}</td><td class="n">${t.z>0?'+':''}${t.z}</td></tr>`;h+=`</table></div>`}
 h+=`<div class="tabs"><button id="o1" ${order==='doc'?'disabled':''}>document order</button><button id="o2" ${order==='score'?'disabled':''}>most AI-like first</button></div>`;
 let ps=d.paragraphs.map((p,i)=>({...p,i}));if(order==='score')ps.sort((a,b)=>b.score-a.score);
 const thr=d.sentence_threshold;
 for(const p of ps){const g=p.signals;const body=p.sentences.map(x=>`<span class="s ${x.score>=thr?'hi':(x.score>=d.sentence_threshold_5?'mid':'')}">${esc(x.text)}</span>`).join(' ');
  h+=`<div class="p ${p.label}"><div class="meta"><b>${p.label}</b> · score ${p.score.toFixed(2)} · paragraph ${p.i+1} · ${Object.entries(g).map(([k,v])=>k+' '+v).join(' · ')}</div>${body}</div>`}
 $('#out').innerHTML=h;$('#o1').onclick=()=>{order='doc';render()};$('#o2').onclick=()=>{order='score';render()}}
$('#go').onclick=async()=>{const text=$('#t').value;if(!text.trim())return;$('#st').textContent='Analyzing (the first run loads the models, about a minute)...';$('#out').innerHTML='';
 try{const r=await fetch('api/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});D=await r.json();render();$('#st').textContent=''}catch(e){$('#st').textContent='Error: '+e}};
</script></body></html>"""

if __name__ == "__main__":
    threading.Thread(target=_reaper, daemon=True).start()
    uvicorn.run(app, host=os.environ.get("AIDET_HOST", "127.0.0.1"), port=int(os.environ.get("AIDET_PORT", "8790")))
