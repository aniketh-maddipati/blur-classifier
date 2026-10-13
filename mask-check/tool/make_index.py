"""Build index.html from results.csv (run from mask-check/)."""
import csv, html, os

all_rows = list(csv.DictReader(open("results.csv")))
# only sheets that are present; sheets with recognisable people are kept out of the repo
rows = [r for r in all_rows if os.path.exists(f"sheets/{r['id']}.jpg")]
cards = []
for r in rows:
    cap = (f"{r['kind']} · prompt: {r['prompt_src']} · person {r['person_ms']} ms · subject {r['subject_ms']} ms "
           f"({r['subject_instances'] or 0} inst.) · SAM 2.1 {r['sam_total_ms']} ms (iou {r['sam_score']})")
    cards.append(f'<figure data-kind="{r["kind"]}"><a href="sheets/{r["id"]}.jpg"><img loading="lazy" src="sheets/{r["id"]}.jpg" alt="{r["id"]}"></a>'
                 f'<figcaption><b>{html.escape(r["id"])}</b><br>{html.escape(cap)}</figcaption></figure>')
kinds = sorted({r["kind"] for r in rows})
buttons = '<button data-k="">all</button>' + "".join(f'<button data-k="{k}">{k}</button>' for k in kinds)
open("index.html", "w").write(f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mask check: Apple Vision vs SAM 2.1</title>
<style>
body{{background:#111;color:#ddd;font:14px -apple-system,system-ui,sans-serif;margin:16px}}
a{{color:#8cf}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px}}
figure{{margin:0;background:#1b1b1b;border-radius:6px;overflow:hidden}} img{{width:100%;display:block}}
figcaption{{padding:8px 10px;color:#aaa;font-size:12px}} figcaption b{{color:#eee}}
button{{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:4px 10px;margin:0 6px 12px 0;cursor:pointer}}
</style>
<h1>Mask check: Apple Vision vs SAM 2.1 small (Core ML)</h1>
<p>{len(all_rows)} Sony ARWs, decoded to 2048 px; {len(rows)} sheets are shown here (the rest show recognisable people or a number plate and are kept local). Each sheet: original · Vision person · Vision subject / attention saliency · SAM 2.1 best-score mask · SAM 2.1 largest mask.
The yellow ring is the single SAM prompt point (face centre if a face was found, else saliency peak). See <a href="README.md">README.md</a> and <a href="results.csv">results.csv</a>.</p>
<div>{buttons}</div><div class="grid">{''.join(cards)}</div>
<script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.querySelectorAll('figure').forEach(f=>f.style.display=(!b.dataset.k||f.dataset.kind===b.dataset.k)?'':'none'))</script>
""")
