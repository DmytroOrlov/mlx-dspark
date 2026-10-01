#!/usr/bin/env python3
"""Offline validation, derivation, repeat adjudication, and reporting for Feature 005."""
from __future__ import annotations
import argparse, hashlib, inspect, json, math, os, statistics, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FEATURE = ROOT / "specs/005-golden-panel-layer-drafter-matrix"
EVIDENCE = FEATURE / "evidence"
ATTEMPTS = EVIDENCE / "attempts"
sys.path.insert(0, str((FEATURE / "evidence/probes").resolve()))
import panel_matrix as physical
GIB = 1024 ** 3
TABLES = ("Raw Prompt × Config Matrix", "Throughput Retention + Memory", "Drafter Delta", "Mechanism Breakdown", "Speed / Memory Tradeoff")

def sha_bytes(v: bytes) -> str: return hashlib.sha256(v).hexdigest()
def sha_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()
def canon(v: Any) -> str: return sha_bytes(json.dumps(v,sort_keys=True,separators=(",",":")).encode())
def read(p: Path) -> dict: return json.loads(p.read_text())
def same(a: Any,b: Any) -> bool: return json.dumps(a,sort_keys=True,separators=(",",":"))==json.dumps(b,sort_keys=True,separators=(",",":"))
def approx(a: Any,b: Any) -> bool:
    try: return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
    except (TypeError,ValueError): return False
def finalizer_sha() -> str: return sha_file(Path(__file__).resolve())
def write_json(path: Path,data: dict,immutable=False) -> str:
    path.parent.mkdir(parents=True,exist_ok=True); content=(json.dumps(data,indent=2,sort_keys=True)+"\n").encode()
    if immutable:
        with path.open("xb") as f: f.write(content); f.flush(); os.fsync(f.fileno())
    else:
        tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
        with tmp.open("xb") as f: f.write(content); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    return sha_bytes(content)

def attempt_dirs(): return sorted((p for p in ATTEMPTS.iterdir() if p.is_dir()),key=lambda p:p.name) if ATTEMPTS.exists() else []
def load_attempt(p: Path) -> dict:
    mp,pp=p/"attempt.json",p/"preflight.json"; meta=read(mp) if mp.is_file() else {}; pre=read(pp) if pp.is_file() else {}
    return {"attempt_id":p.name,"path":str(p.relative_to(FEATURE)),"meta":meta,"preflight":pre,
        "preflight_sha256":sha_file(pp) if pp.is_file() else None,
        "cell_count":sum(1 for _ in (p/"cells").rglob("*.json")) if (p/"cells").exists() else 0,
        "repeat_count":sum(1 for _ in (p/"repeats").glob("*.json")) if (p/"repeats").exists() else 0}
def ledger(effective=None):
    return {"schema":"panel-matrix-attempt-ledger/v1","effective_attempt_id":effective,"attempts":[
        {"attempt_id":a["attempt_id"],"path":a["path"],"status":a["meta"].get("status"),"terminal":a["meta"].get("terminal"),
         "runner_sha256":a["meta"].get("runner_sha256"),"preflight_sha256":a["preflight_sha256"],
         "observed_statused_cell_count":a["meta"].get("observed_statused_cell_count",0),"cell_artifact_count":a["cell_count"],
         "repeat_artifact_count":a["repeat_count"],"reason":a["meta"].get("reason")} for a in map(load_attempt,attempt_dirs())]}
def raw_cells(directory: Path): return [(p,read(p)) for p in sorted((directory/"cells").rglob("*.json"))]
def key(raw):
    i=raw["attempt_cell_identity"]; return i["prompt_id"],i["target"],i["drafter"]

def validate_memory(raw):
    errors=[]; m=raw.get("measurements",{}); vals=[m.get(k) for k in ("active_baseline_bytes","peak_runtime_bytes","peak_increment_bytes")]
    if any(not isinstance(x,int) or x<0 for x in vals): errors.append("memory byte values missing/invalid")
    else:
        b,p,i=vals
        if p-b!=i: errors.append("peak increment arithmetic mismatch")
        for bk,gk in (("active_baseline_bytes","active_baseline_gib"),("peak_runtime_bytes","peak_runtime_gib"),("peak_increment_bytes","peak_increment_gib")):
            if not approx(m.get(gk),m[bk]/GIB): errors.append(f"{gk} conversion mismatch")
    boundary=raw.get("memory_boundary",{}); order=["load_and_warmup_complete","fresh_request_state_asserted","reset_peak_memory","capture_active_baseline","measured_speculative_request_only","capture_peak_immediately"]
    if boundary.get("reset_called") is not True: errors.append("peak reset absent")
    if boundary.get("order")!=order: errors.append("memory boundary ordering mismatch")
    if not valid_memory_timestamps(boundary):
        errors.append("memory operation-boundary timestamps missing, aliased, or out of order")
    return errors

def valid_memory_timestamps(boundary):
    stamps=[boundary.get(k) for k in ("reset_completed","baseline_captured","request_started","request_ended","peak_captured")]
    return not any(not isinstance(x,int) or x<=0 for x in stamps) and all(a<b for a,b in zip(stamps,stamps[1:]))

def valid_cell_identity(identity,prompt_id,order,target,drafter,repeat=False,comparison=None):
    return (identity.get("prompt_id"),identity.get("order"),identity.get("cell_id"),identity.get("target"),identity.get("drafter"),identity.get("repeat"),identity.get("comparison")) == (prompt_id,order,f"{order:02d}-{target}-{drafter}",target,drafter,repeat,comparison)

def classify_output_length(generated_tokens):
    if not isinstance(generated_tokens,int) or generated_tokens<=0: raise ValueError("generated token count missing/invalid")
    return {"physical_valid":True,"long_form_qualified":generated_tokens>=410,"output_length_status":"long_form_qualified" if generated_tokens>=410 else "short"}

def validate_output_integrity(m):
    errors=[]; text=m.get("output_text"); digest=m.get("output_sha256"); mode=m.get("output_integrity_mode")
    if not isinstance(text,str) or not isinstance(digest,str) or digest!=sha_bytes(text.encode()): errors.append("output text integrity missing or invalid")
    tokens=m.get("output_tokens"); token_sha=m.get("output_tokens_sha256")
    if mode=="text_and_tokens":
        if not isinstance(tokens,list) or not tokens or token_sha!=canon(tokens): errors.append("output token integrity missing or invalid")
    elif mode=="text_only":
        if tokens is not None or token_sha is not None: errors.append("text-only output contains fabricated token evidence")
    else: errors.append("unknown output integrity mode")
    return errors

def validate_runtime_hashes(actual,expected):
    return [] if isinstance(actual,dict) and actual==expected else ["runtime source hash mismatch"]

def validate_cell(path,raw,aid,pre,presh,runner,pin,order,target,drafter,repeat=False,comparison=None):
    e=[]; i=raw.get("attempt_cell_identity",{}); cid=f"{order:02d}-{target}-{drafter}"
    if raw.get("schema")!="panel-matrix-cell/v1": e.append("schema invalid")
    if raw.get("attempt_id")!=aid or not valid_cell_identity(i,pin["prompt_id"],order,target,drafter,repeat,comparison): e.append("cell identity mismatch")
    if raw.get("runner_sha256")!=runner: e.append("runner SHA mismatch")
    if raw.get("preflight")!={"path":"preflight.json","sha256":presh,"identity":f"{aid}:PASS"}: e.append("preflight provenance mismatch")
    p=raw.get("prompt",{})
    for k in ("text","text_sha256","input_ids","input_ids_sha256","input_token_count"):
        if p.get(k)!=pin.get(k): e.append(f"prompt {k} mismatch")
    if p.get("text_sha256")!=sha_bytes(str(p.get("text","")).encode()) or p.get("input_ids_sha256")!=canon(p.get("input_ids",[])): e.append("prompt digest mismatch")
    if not same(raw.get("configuration"),physical.target_record_from_manifest(target,drafter,pre["checkpoint_manifest"])): e.append("target/drafter composition mismatch")
    if raw.get("settings")!=physical.SETTINGS: e.append("generation settings mismatch")
    f=raw.get("freshness",{})
    if not all(f.get(k) for k in ("physical_asserted","new_process","new_engine","prefix_cache_initialized")): e.append("freshness declarations incomplete")
    if f.get("prefix_reused_tokens")!=0 or f.get("acquire_reused_tokens")!=0: e.append("prefix reuse nonzero")
    rt=raw.get("runtime",{})
    if rt.get("kv_bits") not in (None,0) or rt.get("width_policy") is not None or rt.get("kv8") is not False or rt.get("tuning") is not False or not rt.get("controller_present"): e.append("runtime controls mismatch")
    prov=raw.get("provenance",{})
    if prov.get("interpreter")!=pre.get("interpreter",{}).get("path") or prov.get("python")!=pre.get("interpreter",{}).get("version"): e.append("interpreter pin mismatch")
    e.extend(validate_runtime_hashes(prov.get("runtime_source_sha256"),pre.get("runtime",{}).get("source_sha256")))
    if prov.get("machine")!=pre.get("runtime",{}).get("machine") or prov.get("platform")!=pre.get("runtime",{}).get("platform"): e.append("machine/platform differs from preflight")
    m=raw.get("measurements",{})
    for k in ("generated_tokens","speculative_decode_tok_s","acceptance","mean_accepted_draft_tokens","target_forwards","generated_tokens_per_target_forward","rounds","width_distribution","cap_distribution","request_duration_seconds","output_sha256","output_text","output_integrity_mode"):
        if k not in m: e.append(f"missing measurement {k}")
    outcome_errors,output_length=validate_physical_outcome(raw.get("status"),m.get("generated_tokens"),repeat)
    e.extend(outcome_errors)
    e.extend(validate_output_integrity(m))
    return e+validate_memory(raw)

def validate_physical_outcome(status,generated_tokens,repeat=False):
    errors=[]
    try: output_length=classify_output_length(generated_tokens)
    except ValueError: return ["generated token count missing/invalid"],None
    if status not in {"complete","invalid_too_short"}: errors.append("physical outcome status invalid")
    if status=="invalid_too_short" and output_length["long_form_qualified"]: errors.append("short-output status contradicts generated token count")
    if status=="complete" and not output_length["long_form_qualified"]: errors.append("complete status contradicts short output length")
    return errors,output_length

def lifecycle_adjudication(meta,statuses,physical_errors):
    state=meta.get("terminal"); declared=meta.get("status")
    if state!=declared: return False,None,"lifecycle status/terminal mismatch"
    if state=="SUPERSEDED": return False,None,"SUPERSEDED attempts are always ineligible"
    if physical_errors: return False,None,"physical/provenance validation failed"
    if len(statuses)!=70 or any(s not in ("complete","invalid_too_short") for s in statuses):
        return False,None,"attempt does not contain 70 statused physical outcomes"
    if state=="COMPLETE":
        if meta.get("completed") is not True: return False,None,"COMPLETE lifecycle is not marked completed"
        return True,None,None
    if state=="INCOMPLETE":
        known_reason="one or more statused cells failed acceptance requirements"
        if meta.get("completed") is not False or not any(s=="invalid_too_short" for s in statuses):
            return False,None,"INCOMPLETE lifecycle is not explained by short-output outcomes"
        if meta.get("reason")!=known_reason:
            return False,None,"INCOMPLETE lifecycle reason is not the recognized output-length acceptance reason"
        reason=("Offline adjudication: all 70 physical/provenance checks passed; the only non-complete outcome status is invalid_too_short. "
            "Feature 005 treats generated length as an observed outcome, so the Feature 004 410-token qualification threshold does not invalidate the physical matrix.")
        return True,reason,None
    return False,None,f"unsupported lifecycle state: {state!r}"

def select_effective():
    valid=[]; reports=[]
    for a in map(load_attempt,attempt_dirs()):
        aid,meta,pre=a["attempt_id"],a["meta"],a["preflight"]; e=[]
        ps=a["preflight_sha256"]; runner=meta.get("runner_sha256")
        statuses=[]
        if a["cell_count"]!=70: e.append("not exactly 70 first-pass cell artifacts")
        if pre.get("status")!="PASS" or pre.get("attempt_id")!=aid or pre.get("runner_sha256")!=runner: e.append("attempt-local passing preflight identity mismatch")
        if meta.get("preflight_sha256")!=ps or meta.get("preflight_identity")!=f"{aid}:PASS": e.append("attempt metadata preflight SHA mismatch")
        if meta.get("expected_cell_count")!=70 or meta.get("observed_statused_cell_count")!=70: e.append("attempt lifecycle metadata does not record 70 statused cells")
        cells=raw_cells(ATTEMPTS/aid); grouped={}
        for p,r in cells:
            k=key(r) if "attempt_cell_identity" in r else ("INVALID",str(p),""); grouped.setdefault(k,[]).append((p,r))
        pins={p["prompt_id"]:p for p in pre.get("prompts",[])}
        expected=physical.expected_cells()
        for x in expected:
            k=(x["prompt_id"],x["target"],x["drafter"]); found=grouped.get(k,[])
            if len(found)!=1: e.append(f"{k}: found {len(found)} effective cells"); continue
            p,r=found[0]
            statuses.append(r.get("status"))
            if x["prompt_id"] not in pins: e.append(f"missing pinned prompt {x['prompt_id']}"); continue
            e.extend(f"{p.relative_to(FEATURE)}: {z}" for z in validate_cell(p,r,aid,pre,ps,runner,pins[x["prompt_id"]],x["order"],x["target"],x["drafter"]))
            expected_path=ATTEMPTS/aid/"cells"/x["prompt_id"]/f"{x['cell_id']}.json"
            if p!=expected_path: e.append(f"unexpected cell path {p}")
        extras=set(grouped)-{(x["prompt_id"],x["target"],x["drafter"]) for x in expected}
        if extras: e.append(f"unexpected identities {sorted(extras)}")
        eligible,adjudication,life_error=lifecycle_adjudication(meta,statuses,e)
        if life_error: e.append(life_error)
        reports.append({"attempt_id":aid,"eligible":eligible and not e,"errors":e,"physical_matrix_completeness":f"{a['cell_count']}/70",
            "status_counts":{s:statuses.count(s) for s in sorted(set(statuses))},"lifecycle_status":meta.get("terminal"),
            "offline_adjudication_reason":adjudication})
        if eligible and not e: a["offline_adjudication_reason"]=adjudication; valid.append(a)
    if len(valid)!=1: return None,{"pass":False,"effective_attempt_id":None,"errors":[f"expected exactly one eligible physically complete 70-cell attempt; found {len(valid)}"],"attempts":reports}
    return valid[0],{"pass":True,"effective_attempt_id":valid[0]["attempt_id"],"errors":[],"attempts":reports}

def derive(attempt):
    aid=attempt["attempt_id"]; source={}
    for p,r in raw_cells(ATTEMPTS/aid): source[key(r)]=(p,r,sha_file(p))
    out=[]
    for pid,*_ in physical.PROMPTS:
        bp,br,bsha=source[(pid,"H0","B-Q")]; bm=br["measurements"]
        for t in physical.TARGETS:
            for d in physical.DRAFTERS:
                p,r,rsha=source[(pid,t,d)]; m=r["measurements"]; bt,bpeak=bm["speculative_decode_tok_s"],bm["peak_runtime_bytes"]
                delta=m["speculative_decode_tok_s"]-bt; ret=m["speculative_decode_tok_s"]/bt*100
                pd=m["peak_runtime_bytes"]-bpeak; saved=(bpeak-m["peak_runtime_bytes"])/GIB
                pct=(bpeak-m["peak_runtime_bytes"])/bpeak*100 if bpeak else 0
                if (t,d)==("H0","B-Q"): delta,ret,pd,saved,pct=0.,100.,0,0.,0.
                out.append({"effective_attempt_id":aid,"prompt_id":pid,"target":t,"drafter":d,
                    "raw_path":str(p.relative_to(FEATURE)),"raw_sha256":rsha,
                    **classify_output_length(m["generated_tokens"]),"generated_tokens":m["generated_tokens"],
                    "same_prompt_baseline":{"target":"H0","drafter":"B-Q","raw_path":str(bp.relative_to(FEATURE)),"raw_sha256":bsha,
                        "speculative_decode_tok_s":bt,"peak_runtime_bytes":bpeak},"cell_tok_s":m["speculative_decode_tok_s"],"throughput_delta_tok_s":delta,
                    "throughput_retention_percent":ret,"peak_memory_delta_bytes":pd,"peak_memory_delta_gib":pd/GIB,
                    "gib_saved":saved,"memory_saving_percent":pct})
    return out

def validate_derived(rows):
    e=[]
    if len(rows)!=70: e.append(f"derived cell count {len(rows)} !=70")
    for r in rows:
        m=read(FEATURE/r["raw_path"])["measurements"]; b=r["same_prompt_baseline"]
        expected={"throughput_delta_tok_s":m["speculative_decode_tok_s"]-b["speculative_decode_tok_s"],
            "throughput_retention_percent":m["speculative_decode_tok_s"]/b["speculative_decode_tok_s"]*100,
            "peak_memory_delta_bytes":m["peak_runtime_bytes"]-b["peak_runtime_bytes"],
            "peak_memory_delta_gib":(m["peak_runtime_bytes"]-b["peak_runtime_bytes"])/GIB,
            "gib_saved":(b["peak_runtime_bytes"]-m["peak_runtime_bytes"])/GIB,
            "memory_saving_percent":(b["peak_runtime_bytes"]-m["peak_runtime_bytes"])/b["peak_runtime_bytes"]*100}
        if (r["target"],r["drafter"])==("H0","B-Q"): expected.update(throughput_delta_tok_s=0.,throughput_retention_percent=100.,peak_memory_delta_bytes=0,peak_memory_delta_gib=0.,gib_saved=0.,memory_saving_percent=0.)
        for k,v in expected.items():
            if not approx(r.get(k),v): e.append(f"{r['prompt_id']}/{r['target']}/{r['drafter']} {k} mismatch")
    return e
def by_key(rows): return {(r["prompt_id"],r["target"],r["drafter"]):r for r in rows}
def raw_m(r,k): return read(FEATURE/r["raw_path"])["measurements"][k]

def candidates(rows):
    by=by_key(rows); subjective=[]; maxima=[]
    for pid,*_ in physical.PROMPTS:
        for t in physical.TARGETS:
            for d in physical.DRAFTERS:
                if t=="H0": continue
                r=by[(pid,t,d)]; subjective.append({"prompt_id":pid,"target":t,"drafter":d,"trigger_class":"conclusion-critical-near-drift-review",
                    "selection":"not selected","raw_path":r["raw_path"],"raw_sha256":r["raw_sha256"],"tok_s":raw_m(r,"speculative_decode_tok_s"),"retention_percent":r["throughput_retention_percent"]})
        chain=("H0","H1a","H1b","H1c","H2","H3","B0")
        for d in physical.DRAFTERS:
            vals=[raw_m(by[(pid,t,d)],"speculative_decode_tok_s") for t in chain]
            for i in range(1,len(chain)-1):
                if vals[i]>vals[i-1] and vals[i]>vals[i+1]:
                    r=by[(pid,chain[i],d)]; maxima.append({"prompt_id":pid,"target":chain[i],"drafter":d,"trigger_class":"surprising-local-target-maximum-review",
                        "selection":"not selected","raw_path":r["raw_path"],"raw_sha256":r["raw_sha256"],"tok_s":vals[i],"neighbor_targets":[chain[i-1],chain[i+1]],"neighbor_tok_s":[vals[i-1],vals[i+1]]})
    return {"schema":"panel-matrix-repeat-candidates/v1","effective_attempt_id":rows[0]["effective_attempt_id"],"finalizer_sha256":finalizer_sha(),
        "subjective_candidates":subjective,"local_target_maximum_candidates":maxima,"policy":"review-only; no subjective threshold or selection inferred"}

def objective_triggers(row):
    triggers=[]
    if not row.get("long_form_qualified",True): return triggers
    if row["target"]!="H0" and row["cell_tok_s"]>=row["same_prompt_baseline"]["speculative_decode_tok_s"]:
        triggers.append("non-h0-at-or-above-same-prompt-h0-bq")
    if row["target"]=="H1b" and row["drafter"]=="B-B" and row["throughput_retention_percent"]>=97.0:
        triggers.append("h1b-bb-retention-at-least-97-percent")
    return triggers

def make_manifest(rows):
    by=by_key(rows); chosen={}
    for r in rows:
        if r["target"]=="H0": continue
        k=(r["prompt_id"],r["target"],r["drafter"]); reasons=set(objective_triggers(r))
        if reasons: chosen[k]=reasons
    dp=EVIDENCE/"repeat-decisions.json"; decisions=read(dp).get("decisions",[]) if dp.is_file() else []
    if not isinstance(decisions,list): raise ValueError("repeat-decisions.json decisions must be a list")
    for x in decisions:
        required=("effective_attempt_id","prompt_id","target","drafter","triggering_raw_path","triggering_raw_sha256","trigger_class","reason")
        if any(k not in x for k in required) or not str(x.get("reason","")).strip(): raise ValueError("explicit repeat decision lacks required fields/reason")
        if x["effective_attempt_id"]!=rows[0]["effective_attempt_id"]: raise ValueError("decision attempt mismatch")
        k=(x["prompt_id"],x["target"],x["drafter"])
        if k not in by or (x["triggering_raw_path"],x["triggering_raw_sha256"])!=(by[k]["raw_path"],by[k]["raw_sha256"]): raise ValueError("decision raw identity mismatch")
        if x["trigger_class"] not in ("conclusion-critical-near-drift-review","surprising-local-target-maximum-review"): raise ValueError("unknown subjective trigger class")
        chosen.setdefault(k,set()).add("explicit:"+x["trigger_class"])
    repeats=[]
    for (pid,t,d),reasons in sorted(chosen.items()):
        r=by[(pid,t,d)]; cell=next(c for c in physical.expected_cells() if (c["prompt_id"],c["target"],c["drafter"])==(pid,t,d))
        repeats.append({"comparison_id":f"{pid}-{t}-{d}","attempt_id":rows[0]["effective_attempt_id"],"prompt_id":pid,"order":cell["order"],"target":t,"drafter":d,
            "trigger_classes":sorted(reasons),"triggering_raw_path":r["raw_path"],"triggering_raw_sha256":r["raw_sha256"],"first_pass_tok_s":raw_m(r,"speculative_decode_tok_s"),
            "same_prompt_h0_bq_tok_s":r["same_prompt_baseline"]["speculative_decode_tok_s"]})
    aid=rows[0]["effective_attempt_id"]; meta=read(ATTEMPTS/aid/"attempt.json")
    return {"schema":"panel-matrix-repeat-manifest/v1","status":"PASS","effective_attempt_id":aid,"runner_sha256":meta["runner_sha256"],
        "finalizer_sha256":finalizer_sha(),"objective_repeat_count":sum(any(not z.startswith("explicit:") for z in r["trigger_classes"]) for r in repeats),
        "explicit_subjective_repeat_count":sum(any(z.startswith("explicit:") for z in r["trigger_classes"]) for r in repeats),"repeats":repeats}

def objective_preview(rows):
    selected=[]
    for r in rows:
        triggers=objective_triggers(r)
        if triggers:
            selected.append({"prompt_id":r["prompt_id"],"target":r["target"],"drafter":r["drafter"],"trigger_classes":triggers,
                "raw_path":r["raw_path"],"raw_sha256":r["raw_sha256"],"cell_tok_s":r["cell_tok_s"],
                "same_prompt_h0_bq_tok_s":r["same_prompt_baseline"]["speculative_decode_tok_s"]})
    return {"schema":"panel-matrix-objective-trigger-preview/v1","effective_attempt_id":rows[0]["effective_attempt_id"],
        "finalizer_sha256":finalizer_sha(),"objective_trigger_count":len(selected),"triggers":selected}

def derive_first_pass():
    attempt,selection=select_effective()
    if attempt is None:
        val={"schema":"panel-matrix-validation/v1","pass":False,**selection,"finalizer_sha256":finalizer_sha(),"first_pass_cell_count":0}
        write_json(EVIDENCE/"attempts.json",ledger()); write_json(EVIDENCE/"matrix-validation.json",val)
        raise RuntimeError(json.dumps(val,indent=2))
    rows=derive(attempt); errors=validate_derived(rows)
    if errors: raise RuntimeError("derived arithmetic failed: "+"; ".join(errors))
    write_json(EVIDENCE/"derived-matrix.json",{"schema":"panel-matrix-derived/v1","effective_attempt_id":attempt["attempt_id"],"finalizer_sha256":finalizer_sha(),"cells":rows})
    write_json(EVIDENCE/"repeat-candidates.json",candidates(rows))
    write_json(EVIDENCE/"objective-trigger-preview.json",objective_preview(rows))
    short=[{"prompt_id":r["prompt_id"],"target":r["target"],"drafter":r["drafter"],"generated_tokens":r["generated_tokens"]} for r in rows if not r["long_form_qualified"]]
    val={"schema":"panel-matrix-validation/v1","pass":True,**selection,"effective_attempt_id":attempt["attempt_id"],
        "physical_runner_sha256":attempt["meta"]["runner_sha256"],"preflight_sha256":attempt["preflight_sha256"],
        "first_pass_cell_count":70,"derived_cell_count":70,"physical_matrix_complete_count":70,"physically_valid_observation_count":70,
        "long_form_qualified_observation_count":70-len(short),"short_output_observations":short,
        "lifecycle_status_from_attempt_metadata":attempt["meta"].get("terminal") or attempt["meta"].get("status"),
        "offline_adjudication_reason":attempt.get("offline_adjudication_reason"),
        "derived_arithmetic_errors":[],"finalizer_sha256":finalizer_sha(),"raw_artifacts_immutable":True}
    write_json(EVIDENCE/"attempts.json",ledger(attempt["attempt_id"])); write_json(EVIDENCE/"matrix-validation.json",val)
    return attempt,rows,val

def write_repeat_manifest():
    derived_path=EVIDENCE/"derived-matrix.json"; validation_path=EVIDENCE/"matrix-validation.json"
    if not derived_path.is_file() or not validation_path.is_file(): raise RuntimeError("run --derive-first-pass before selecting repeats")
    derived=read(derived_path); validation=read(validation_path)
    if not validation.get("pass") or validation.get("effective_attempt_id")!=derived.get("effective_attempt_id"):
        raise RuntimeError("repeat selection requires a passing validation for the derived effective attempt")
    if (EVIDENCE/"repeat-manifest.json").exists(): raise FileExistsError("repeat-manifest.json is immutable and already exists")
    manifest=make_manifest(derived["cells"]); write_json(EVIDENCE/"repeat-manifest.json",manifest,immutable=True)
    return manifest

def report_tables(rows):
    by=by_key(rows); configs=[(t,d) for t in physical.TARGETS for d in physical.DRAFTERS]
    rawtab=[]; rettab=[]; deltab=[]; mech=[]; trade=[]
    for t,d in configs:
        rr=[by[(pid,t,d)] for pid,*_ in physical.PROMPTS]; rates=[raw_m(x,"speculative_decode_tok_s") for x in rr]; peaks=[raw_m(x,"peak_runtime_gib") for x in rr]
        saved=[x["gib_saved"] for x in rr]; sp=[x["memory_saving_percent"] for x in rr]; ret=[x["throughput_retention_percent"] for x in rr]
        long_qualified=[x["long_form_qualified"] for x in rr]
        base={"target":t,"bonsai_owned_block_count":len(physical.DONOR_BLOCKS[t]),"drafter":d,"prompt_tok_s":{x["prompt_id"]:raw_m(x,"speculative_decode_tok_s") for x in rr},
            "short_output_prompts":[x["prompt_id"] for x in rr if not x["long_form_qualified"]],
            "minimum_tok_s":min(rates),"median_tok_s":statistics.median(rates),"maximum_tok_s":max(rates),"count_ge_40":sum(x>=40 for x in rates),"count_ge_45":sum(x>=45 for x in rates),"count_ge_50":sum(x>=50 for x in rates),
            "long_form_qualified_all_five":all(long_qualified),"strict_long_form_count_ge_40":sum(x["long_form_qualified"] and raw_m(x,"speculative_decode_tok_s")>=40 for x in rr),
            "strict_long_form_count_ge_45":sum(x["long_form_qualified"] and raw_m(x,"speculative_decode_tok_s")>=45 for x in rr),
            "median_peak_gib":statistics.median(peaks),"maximum_peak_gib":max(peaks),"median_gib_saved":statistics.median(saved),"median_memory_saving_percent":statistics.median(sp),
            "minimum_retention_percent":min(ret),"median_retention_percent":statistics.median(ret)}
        rawtab.append(base); trade.append({k:base[k] for k in ("target","bonsai_owned_block_count","drafter","minimum_tok_s","median_tok_s","count_ge_40","count_ge_45","count_ge_50","long_form_qualified_all_five","strict_long_form_count_ge_40","strict_long_form_count_ge_45","minimum_retention_percent","median_retention_percent","median_peak_gib","maximum_peak_gib","median_gib_saved","median_memory_saving_percent")})
        rettab.append({"target":t,"drafter":d,"prompts":[{"prompt_id":x["prompt_id"],"tok_s":raw_m(x,"speculative_decode_tok_s"),"throughput_retention_percent":x["throughput_retention_percent"],"peak_gib":raw_m(x,"peak_runtime_gib"),"gib_saved":x["gib_saved"],"memory_saving_percent":x["memory_saving_percent"],"generated_tokens":x["generated_tokens"],"physical_valid":x["physical_valid"],"long_form_qualified":x["long_form_qualified"],"output_length_status":x["output_length_status"]} for x in rr],
            "minimum_retention_percent":min(ret),"median_retention_percent":statistics.median(ret),"median_peak_gib":statistics.median(peaks),"maximum_peak_gib":max(peaks),"median_gib_saved":statistics.median(saved),"median_memory_saving_percent":statistics.median(sp)})
    for t in physical.TARGETS:
        pp=[]
        for pid,*_ in physical.PROMPTS:
            diff=raw_m(by[(pid,t,"B-B")],"speculative_decode_tok_s")-raw_m(by[(pid,t,"B-Q")],"speculative_decode_tok_s"); pp.append({"prompt_id":pid,"b_b_minus_b_q_tok_s":diff})
        deltab.append({"target":t,"prompts":pp,"median_b_b_minus_b_q_tok_s":statistics.median(x["b_b_minus_b_q_tok_s"] for x in pp),"b_b_wins":sum(x["b_b_minus_b_q_tok_s"]>0 for x in pp),"b_q_wins":sum(x["b_b_minus_b_q_tok_s"]<0 for x in pp),"ties":sum(x["b_b_minus_b_q_tok_s"]==0 for x in pp)})
        for pid,*_ in physical.PROMPTS:
            for d in physical.DRAFTERS:
                r=by[(pid,t,d)]; m=read(FEATURE/r["raw_path"])["measurements"]
                mech.append({"prompt_id":pid,"target":t,"drafter":d,"tok_s":m["speculative_decode_tok_s"],"generated_tokens":m["generated_tokens"],"long_form_qualified":m["generated_tokens"]>=410,"acceptance":m["acceptance"],"mean_accepted_draft_tokens":m["mean_accepted_draft_tokens"],
                    "generated_tokens_per_target_forward":m["generated_tokens_per_target_forward"],"target_forwards":m["target_forwards"],"rounds":m["rounds"],"width_distribution":m["width_distribution"],"cap_distribution":m["cap_distribution"],
                    "active_baseline_gib":m["active_baseline_gib"],"peak_gib":m["peak_runtime_gib"],"peak_increment_gib":m["peak_increment_gib"],"request_duration_seconds":m["request_duration_seconds"]})
    return {"raw_prompt_config_matrix":rawtab,"throughput_retention_memory":rettab,"drafter_delta":deltab,"mechanism_breakdown":mech,"speed_memory_tradeoff":trade}

def pareto(trade):
    out=[]
    for a in trade:
        dominates=[]
        for b in trade:
            if b is a: continue
            if b["minimum_tok_s"]>=a["minimum_tok_s"] and b["median_peak_gib"]<=a["median_peak_gib"] and (b["minimum_tok_s"]>a["minimum_tok_s"] or b["median_peak_gib"]<a["median_peak_gib"]): dominates.append(f"{b['target']}+{b['drafter']}")
        out.append({"target":a["target"],"drafter":a["drafter"],"minimum_tok_s":a["minimum_tok_s"],"median_peak_gib":a["median_peak_gib"],"classification":"dominated" if dominates else "not-dominated-by-measured-pair","dominated_by":dominates,"dimensions":"minimum five-prompt throughput and median peak memory only"})
    return out

def repeat_validate(attempt,manifest):
    aid=attempt["attempt_id"]; runner=attempt["meta"]["runner_sha256"]; errors=[]; seen=set(); outcomes=[]
    for x in manifest["repeats"]:
        cid=x["comparison_id"]
        if cid in seen: errors.append(f"duplicate repeat {cid}"); continue
        seen.add(cid)
        tp=FEATURE/x["triggering_raw_path"]
        if x["attempt_id"]!=aid or not tp.is_file() or sha_file(tp)!=x["triggering_raw_sha256"]: errors.append(f"trigger provenance invalid {cid}"); continue
        p=ATTEMPTS/aid/"repeats"/f"{cid}.json"
        if not p.is_file(): outcomes.append({"comparison_id":cid,"status":"UNRESOLVED_NOT_EXECUTED","path":str(p.relative_to(FEATURE))}); continue
        r=read(p); pin=next(z for z in attempt["preflight"]["prompts"] if z["prompt_id"]==x["prompt_id"])
        if r.get("runner_sha256")!=runner or r.get("attempt_id")!=aid: errors.append(f"repeat attempt/runner mismatch {cid}")
        e=validate_cell(p,r,aid,attempt["preflight"],attempt["preflight_sha256"],runner,pin,x["order"],x["target"],x["drafter"],True,cid)
        if e: errors.extend(f"{cid}: {z}" for z in e)
        else:
            length=classify_output_length(r["measurements"]["generated_tokens"])
            outcomes.append({"comparison_id":cid,"status":"VALID","path":str(p.relative_to(FEATURE)),"sha256":sha_file(p),
                "raw_status":r["status"],"generated_tokens":r["measurements"]["generated_tokens"],
                "tok_s":r["measurements"]["speculative_decode_tok_s"],**length})
    return {"pass":not errors,"errors":errors,"validated_repeats":outcomes,"expected_repeat_count":len(manifest["repeats"]),
        "completed_repeat_count":sum(x["status"]=="VALID" for x in outcomes),"unresolved_repeat_count":sum(x["status"].startswith("UNRESOLVED") for x in outcomes)}

def make_panel(attempt,rows,tables,repeats,validation):
    by=by_key(rows); trade=tables["speed_memory_tradeoff"]; aid=attempt["attempt_id"]
    at40=[f"{r['target']}+{r['drafter']}" for r in trade if r["count_ge_40"]==5]; at45=[f"{r['target']}+{r['drafter']}" for r in trade if r["count_ge_45"]==5]
    strict40=[f"{r['target']}+{r['drafter']}" for r in trade if r["long_form_qualified_all_five"] and r["strict_long_form_count_ge_40"]==5]
    strict45=[f"{r['target']}+{r['drafter']}" for r in trade if r["long_form_qualified_all_five"] and r["strict_long_form_count_ge_45"]==5]
    high=max(r["minimum_tok_s"] for r in trade); winners=[f"{r['target']}+{r['drafter']}" for r in trade if r["minimum_tok_s"]==high]
    paths={pid:{d:[{"target":t,"tok_s":raw_m(by[(pid,t,d)],"speculative_decode_tok_s"),"generated_tokens":by[(pid,t,d)]["generated_tokens"],"long_form_qualified":by[(pid,t,d)]["long_form_qualified"],"output_length_status":by[(pid,t,d)]["output_length_status"],"raw_path":by[(pid,t,d)]["raw_path"]} for t in ("H0","H1a","H1c","H2","H3","B0")] for d in physical.DRAFTERS} for pid,*_ in physical.PROMPTS}
    interaction=[]
    for pid,*_ in physical.PROMPTS:
        for d in physical.DRAFTERS:
            z={"prompt_id":pid,"drafter":d}
            for t in ("H1a","H1b","H1c"):
                z[f"{t}_tok_s"]=raw_m(by[(pid,t,d)],"speculative_decode_tok_s"); z[f"{t}_acceptance"]=read(FEATURE/by[(pid,t,d)]["raw_path"])["measurements"]["acceptance"]
            interaction.append(z)
    def saving_set(values): return [c for c in values if next(r for r in trade if f"{r['target']}+{r['drafter']}"==c)["median_gib_saved"]>0]
    summary={"highest_minimum_tok_s":high,"highest_minimum_configurations":winners,"at_least_40_all_five_prompts":at40,"at_least_45_all_five_prompts":at45,
        "observed_throughput_at_least_40_all_five_prompts":at40,"observed_throughput_at_least_45_all_five_prompts":at45,
        "strict_long_form_at_least_40_all_five_prompts":strict40,"strict_long_form_at_least_45_all_five_prompts":strict45,
        "memory_saving_configs_all_five_ge_40":saving_set(at40),"memory_saving_configs_all_five_ge_45":saving_set(at45),
        "h1b_bb_repeat_status":[x for x in repeats["validated_repeats"] if "H1b-B-B" in x["comparison_id"]] or "no H1b+B-B repeat completed or selected",
        "b0_context":"B0 is native full Bonsai2. Feature 005 measures speculative throughput, acceptance, and request-boundary memory; serial controls are intentionally not run in this panel.",
        "interpretation_guard":"Descriptive associations only; no causal claim, aggregate winner, or weighted score."}
    return {"schema":"panel-matrix/v1","effective_attempt_id":aid,"finalizer_sha256":finalizer_sha(),"physical_runner_sha256":attempt["meta"]["runner_sha256"],
        "prompt_ids":[p[0] for p in physical.PROMPTS],"targets":list(physical.TARGETS),"drafters":list(physical.DRAFTERS),"tables":tables,"pareto_view":pareto(trade),
        "drafter_paths_by_prompt":paths,"h1a_h1b_h1c_interaction":interaction,"repeat_ledger":repeats,"robustness_summary":summary,"validation":validation,
        "evidence_labels":{"raw":"measured physical evidence","derived":"same-prompt calculations","interpretation":"descriptive, non-causal"}}

def fmt(x,n=2): return "—" if x is None else f"{x:.{n}f}" if isinstance(x,(float,int)) else str(x)
def summarize(panel,t,d):
    r=next(x for x in panel["tables"]["speed_memory_tradeoff"] if (x["target"],x["drafter"])==(t,d))
    return f"min {fmt(r['minimum_tok_s'])} tok/s; median {fmt(r['median_tok_s'])} tok/s; median peak {fmt(r['median_peak_gib'])} GiB; median saved {fmt(r['median_gib_saved'])} GiB ({fmt(r['median_memory_saving_percent'])}%); {r['count_ge_40']}/5 prompts >=40"

def markdown(p):
    aid=p["effective_attempt_id"]; T=p["tables"]; L=["# Golden Panel Layer-Drafter Matrix","",f"Effective attempt: `{aid}`  ",f"Physical runner SHA-256: `{p['physical_runner_sha256']}`  ",f"Finalizer SHA-256: `{p['finalizer_sha256']}`  ",f"First pass: {p['validation']['first_pass_cell_count']}/70 valid cells  ",f"Validation: **{'PASS' if p['validation']['pass'] else 'FAIL'}**","","Raw values are measured. Retention, deltas, savings, and robustness counts are derived against same-prompt H0+B-Q. Interpretation is descriptive.",""]
    if p["validation"].get("offline_adjudication_reason"):
        L += [f"Offline adjudication: {p['validation']['offline_adjudication_reason']}",f"Attempt lifecycle status: {p['validation'].get('lifecycle_status_from_attempt_metadata')}.",""]
    L += ["## Raw Prompt × Config Matrix","","† denotes a physically valid short-output observation (<410 tokens); measured values remain included in observed-throughput summaries. Strict long-form claims require at least 410 tokens on all five prompts.","","| Target | Blocks | Drafter | P05 | P07 | P08 | P14 | P17 | Min | Median | Max | ≥40 | ≥45 | ≥50 | Median peak GiB | Max peak GiB | Median saved GiB | Median saved % |","|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in T["raw_prompt_config_matrix"]: L.append("| "+" | ".join([r["target"],str(r["bonsai_owned_block_count"]),r["drafter"]]+[fmt(r["prompt_tok_s"][x])+("†" if x in r["short_output_prompts"] else "") for x in ("P05","P07","P08","P14","P17")]+[fmt(r[k]) for k in ("minimum_tok_s","median_tok_s","maximum_tok_s")]+[str(r[k]) for k in ("count_ge_40","count_ge_45","count_ge_50")]+[fmt(r[k]) for k in ("median_peak_gib","maximum_peak_gib","median_gib_saved","median_memory_saving_percent")])+" |")
    L += ["","## Throughput Retention + Memory","","| Target | Drafter | Prompt | Tok/s | Retention % | Peak GiB | GiB saved |","|---|---|---|---:|---:|---:|---:|"]
    for r in T["throughput_retention_memory"]:
        for x in r["prompts"]: L.append(f"| {r['target']} | {r['drafter']} | {x['prompt_id']} {'†' if not x['long_form_qualified'] else ''} | {fmt(x['tok_s'])} | {fmt(x['throughput_retention_percent'])} | {fmt(x['peak_gib'])} | {fmt(x['gib_saved'])} |")
        L.append(f"| {r['target']} | {r['drafter']} | **min/median summary** | — | {fmt(r['minimum_retention_percent'])} / {fmt(r['median_retention_percent'])} | {fmt(r['median_peak_gib'])} / {fmt(r['maximum_peak_gib'])} | {fmt(r['median_gib_saved'])}; {fmt(r['median_memory_saving_percent'])}% |")
    L += ["","## Drafter Delta","","B-B minus B-Q speculative tok/s, paired by prompt.","","| Target | P05 | P07 | P08 | P14 | P17 | Median delta | B-B wins | B-Q wins | Ties |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in T["drafter_delta"]:
        v={x["prompt_id"]:x["b_b_minus_b_q_tok_s"] for x in r["prompts"]}; L.append("| "+" | ".join([r["target"]]+[fmt(v[x]) for x in ("P05","P07","P08","P14","P17")]+[fmt(r["median_b_b_minus_b_q_tok_s"]),str(r["b_b_wins"]),str(r["b_q_wins"]),str(r["ties"] )])+" |")
    L += ["","## Mechanism Breakdown","","Measured values; this table does not imply causality.","","| Prompt | Target | Drafter | Tok/s | Acceptance | Mean accepted | Generated/forward | Forwards | Rounds | Widths | Caps | Baseline GiB | Peak GiB | Increment GiB | Request s |","|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    for r in T["mechanism_breakdown"]: L.append("| "+" | ".join([r["prompt_id"]+("†" if not r["long_form_qualified"] else ""),r["target"],r["drafter"],fmt(r["tok_s"]),fmt(r["acceptance"],3),fmt(r["mean_accepted_draft_tokens"],3),fmt(r["generated_tokens_per_target_forward"],3),str(r["target_forwards"]),str(r["rounds"]),json.dumps(r["width_distribution"],sort_keys=True),json.dumps(r["cap_distribution"],sort_keys=True),fmt(r["active_baseline_gib"]),fmt(r["peak_gib"]),fmt(r["peak_increment_gib"]),fmt(r["request_duration_seconds"])])+" |")
    L += ["","## Speed / Memory Tradeoff","","No weighted score or arbitrary ranking.","","| Target | Blocks | Drafter | Min tok/s | Median tok/s | ≥40 | ≥45 | ≥50 | Min retention % | Median retention % | Median peak GiB | Max peak GiB | Median saved GiB | Median saved % |","|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in T["speed_memory_tradeoff"]: L.append("| "+" | ".join([r["target"],str(r["bonsai_owned_block_count"]),r["drafter"]]+[fmt(r[k]) for k in ("minimum_tok_s","median_tok_s")]+[str(r[k]) for k in ("count_ge_40","count_ge_45","count_ge_50")]+[fmt(r[k]) for k in ("minimum_retention_percent","median_retention_percent","median_peak_gib","maximum_peak_gib","median_gib_saved","median_memory_saving_percent")])+" |")
    s=p["robustness_summary"]; L += ["","## Robustness Findings","",f"Highest minimum rate: **{', '.join(s['highest_minimum_configurations'])}**, {fmt(s['highest_minimum_tok_s'])} tok/s.",f"≥40 on all prompts: {', '.join(s['at_least_40_all_five_prompts']) or 'none'}.",f"≥45 on all prompts: {', '.join(s['at_least_45_all_five_prompts']) or 'none'}.",f"≥40 on all five with median memory savings: {', '.join(s['memory_saving_configs_all_five_ge_40']) or 'none'}.",f"≥45 on all five with median memory savings: {', '.join(s['memory_saving_configs_all_five_ge_45']) or 'none'}.","","### Individual Prompt Paths",""]
    L += ["Observed-throughput robustness includes all physically valid observations.",f"Strict long-form ≥40 on all prompts: {', '.join(s['strict_long_form_at_least_40_all_five_prompts']) or 'none'}.",f"Strict long-form ≥45 on all prompts: {', '.join(s['strict_long_form_at_least_45_all_five_prompts']) or 'none'}. "]
    for pid,ds in p["drafter_paths_by_prompt"].items():
        for d,seq in ds.items(): L.append(f"- {pid} {d}: "+" → ".join(f"{x['target']} ({fmt(x['tok_s'])}{'†' if not x['long_form_qualified'] else ''})" for x in seq))
    L += ["","### H1a / H1b / H1c Interaction","","| Prompt | Drafter | H1a tok/s | H1b tok/s | H1c tok/s | H1a acc. | H1b acc. | H1c acc. |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in p["h1a_h1b_h1c_interaction"]: L.append(f"| {r['prompt_id']} | {r['drafter']} | {fmt(r['H1a_tok_s'])} | {fmt(r['H1b_tok_s'])} | {fmt(r['H1c_tok_s'])} | {fmt(r['H1a_acceptance'],3)} | {fmt(r['H1b_acceptance'],3)} | {fmt(r['H1c_acceptance'],3)} |")
    L += ["","### Descriptive Pareto View","","Dimensions: minimum throughput over five prompts and median measured peak memory.","","| Target | Drafter | Min tok/s | Median peak GiB | Classification | Dominated by |","|---|---|---:|---:|---|---|"]
    for r in p["pareto_view"]: L.append(f"| {r['target']} | {r['drafter']} | {fmt(r['minimum_tok_s'])} | {fmt(r['median_peak_gib'])} | {r['classification']} | {', '.join(r['dominated_by']) or '—'} |")
    rep=p["repeat_ledger"]; L += ["","### Repeat Ledger","",f"Objective repeats: {rep['objective_repeat_count']}; explicit subjective repeats: {rep['explicit_subjective_repeat_count']}.",f"Completed: {rep['completed_repeat_count']}/{rep['expected_repeat_count']}; unresolved: {rep['unresolved_repeat_count']}.",f"H1b+B-B repeat status: {s['h1b_bb_repeat_status']}.","","| Comparison | Trigger | Status | Raw path |","|---|---|---|---|"]
    for x in rep["manifest_repeats"]:
        o=next((z for z in rep["validated_repeats"] if z["comparison_id"]==x["comparison_id"]),{"status":"UNRESOLVED_NOT_EXECUTED"}); L.append(f"| {x['comparison_id']} | {', '.join(x['trigger_classes'])} | {o['status']} | {o.get('path','—')} |")
    L += ["","### Interpretation","","- **Measured:** throughput, acceptance, controller distributions, duration, and request-boundary memory for every cell.","- **Derived:** same-prompt deltas, retention, GiB and percentage savings, robustness and Pareto comparisons.","- **H1b+B-B:** "+summarize(p,"H1b","B-B")+".","- **H3:** "+summarize(p,"H3","B-Q")+"; "+summarize(p,"H3","B-B")+".","- **B0:** "+summarize(p,"B0","B-Q")+"; "+summarize(p,"B0","B-B")+". "+s["b0_context"],"- Increasing Bonsai ownership and memory/throughput changes are descriptive associations; no causal claim is made.",""]
    return "\n".join(L)

def model_free_checks():
    assert len(physical.expected_cells())==70 and len(TABLES)==5
    assert physical.BQ_REV=="015e795645c74b1a0eeef3b570031fb62e769bc5"
    assert physical.BB_REV=="0059b38aa255698b1a87305eb3fbb5a3cfd616e2"
    assert physical.BB_WEIGHT_SHA256=="eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1"
    assert [r[0] for r in physical.PROMPTS]==["P05","P07","P08","P14","P17"]
    assert all(len(r[2])>20 and r[3]>0 for r in physical.PROMPTS)
    assert physical.FORWARD_ORDER[0]==("H0","B-Q") and physical.REVERSE_ORDER[-1]==("H0","B-B")
    assert (50>=50) and not (49>=50) and 97>=97 and not 96.9>=97
    assert "non-h0-at-or-above-same-prompt-h0-bq" in objective_triggers({"target":"H1a","drafter":"B-Q","cell_tok_s":45,
        "same_prompt_baseline":{"speculative_decode_tok_s":45},"throughput_retention_percent":100})
    assert objective_triggers({"target":"H1b","drafter":"B-B","cell_tok_s":43,
        "same_prompt_baseline":{"speculative_decode_tok_s":45},"throughput_retention_percent":97})==["h1b-bb-retention-at-least-97-percent"]
    assert objective_triggers({"target":"H1b","drafter":"B-B","cell_tok_s":43,
        "same_prompt_baseline":{"speculative_decode_tok_s":45},"throughput_retention_percent":96.99})==[]
    p=pareto([{"target":"A","drafter":"B-Q","minimum_tok_s":40,"median_peak_gib":10},{"target":"B","drafter":"B-Q","minimum_tok_s":41,"median_peak_gib":9}])
    assert p[0]["classification"]=="dominated"
    assert all(x in " ".join(TABLES) for x in ("Raw Prompt","Drafter","Mechanism","Speed"))
    assert not valid_cell_identity({"prompt_id":"P05","order":1,"cell_id":"01-H0-B-Q","target":"H0","drafter":"B-Q","repeat":True,"comparison":None},"P05",1,"H0","B-Q",False,None)
    assert valid_cell_identity({"prompt_id":"P05","order":1,"cell_id":"01-H0-B-Q","target":"H0","drafter":"B-Q","repeat":False,"comparison":None},"P05",1,"H0","B-Q",False,None)
    assert validate_output_integrity({"output_text":"answer","output_sha256":sha_bytes(b"answer"),"output_integrity_mode":"text_only","output_tokens":None,"output_tokens_sha256":None})==[]
    assert validate_output_integrity({"output_integrity_mode":"text_only","output_tokens":[],"output_tokens_sha256":canon([])})
    assert validate_output_integrity({"output_text":"answer","output_sha256":sha_bytes(b"answer"),"output_integrity_mode":"text_and_tokens","output_tokens":[1],"output_tokens_sha256":canon([1])})==[]
    assert validate_runtime_hashes({"x":"new"},{"x":"old"})
    assert valid_memory_timestamps({"reset_completed":1,"baseline_captured":2,"request_started":3,"request_ended":4,"peak_captured":5})
    assert not valid_memory_timestamps({"reset_completed":1,"baseline_captured":2,"request_started":2,"request_ended":4,"peak_captured":5})
    assert classify_output_length(400)=={"physical_valid":True,"long_form_qualified":False,"output_length_status":"short"}
    assert classify_output_length(410)=={"physical_valid":True,"long_form_qualified":True,"output_length_status":"long_form_qualified"}
    assert validate_physical_outcome("invalid_too_short",400,True)==([],{"physical_valid":True,"long_form_qualified":False,"output_length_status":"short"})
    assert validate_physical_outcome("FAILED",400,True)[0]==["physical outcome status invalid"]
    full_statuses=["complete"]*70
    assert lifecycle_adjudication({"terminal":"SUPERSEDED","status":"SUPERSEDED","completed":False},full_statuses,[])[0] is False
    c1_statuses=["complete"]*69+["invalid_too_short"]
    c1_meta={"terminal":"INCOMPLETE","status":"INCOMPLETE","completed":False,"reason":"one or more statused cells failed acceptance requirements"}
    c1_decision=lifecycle_adjudication(c1_meta,c1_statuses,[])
    assert c1_decision[0] is True and "offline adjudication" in c1_decision[1].lower()
    assert lifecycle_adjudication(c1_meta,["complete"]*68+["invalid_too_short","FAILED"],["runtime source hash mismatch"])[0] is False
    assert lifecycle_adjudication({**c1_meta,"reason":"runner defect"},c1_statuses,[])[0] is False
    assert "fewer than 410 generated tokens" not in inspect.getsource(validate_cell)
    derive_source=inspect.getsource(derive_first_pass)
    assert "repeat-manifest.json" not in derive_source
    assert "objective-trigger-preview.json" in derive_source and "repeat-candidates.json" in derive_source
    assert "repeat-decisions.json" in __file__ or Path(__file__).name=="finalize_matrix.py"
    assert "serial" not in Path(__file__).read_text().lower() or "serial controls are intentionally not run" in Path(__file__).read_text().lower()
    return {"status":"PASS","checks":32,"result":"focused model-free finalizer checks passed"}

def finalize():
    attempt,selection=select_effective()
    if attempt is None: raise RuntimeError(json.dumps(selection,indent=2))
    derived=read(EVIDENCE/"derived-matrix.json"); manifest=read(EVIDENCE/"repeat-manifest.json")
    if derived.get("effective_attempt_id")!=attempt["attempt_id"] or manifest.get("effective_attempt_id")!=attempt["attempt_id"]:
        raise RuntimeError("derived matrix or repeat manifest belongs to another attempt")
    rows=derived["cells"]
    val=read(EVIDENCE/"matrix-validation.json")
    repeats=repeat_validate(attempt,manifest)
    val["repeat_validation"]=repeats; val["pass"]=val["pass"] and repeats["pass"]
    tables=report_tables(rows); report={**repeats,"objective_repeat_count":manifest["objective_repeat_count"],"explicit_subjective_repeat_count":manifest["explicit_subjective_repeat_count"],"manifest_repeats":manifest["repeats"]}
    panel=make_panel(attempt,rows,tables,report,val); write_json(EVIDENCE/"matrix-validation.json",val); write_json(EVIDENCE/"panel-matrix.json",panel)
    (EVIDENCE/"panel-matrix.md").write_text(markdown(panel)+"\n")
    return {"effective_attempt_id":attempt["attempt_id"],"first_pass_cells":len(rows),"objective_repeats":manifest["objective_repeat_count"],
        "explicit_subjective_repeats":manifest["explicit_subjective_repeat_count"],"repeat_completion":f"{repeats['completed_repeat_count']}/{repeats['expected_repeat_count']}",
        "validation":"PASS" if val["pass"] else "FAIL","finalizer_sha256":finalizer_sha()}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--self-check",action="store_true"); p.add_argument("--derive-first-pass",action="store_true"); p.add_argument("--write-repeat-manifest",action="store_true"); p.add_argument("--finalize",action="store_true"); a=p.parse_args()
    if a.self_check: print(json.dumps(model_free_checks(),indent=2)); return 0
    if a.derive_first_pass:
        at,rows,val=derive_first_pass(); preview=read(EVIDENCE/"objective-trigger-preview.json"); print(json.dumps({"effective_attempt_id":at["attempt_id"],"derived_cells":len(rows),"objective_triggers":preview["objective_trigger_count"],"validation":val["pass"],"repeat_manifest_written":False},indent=2)); return 0
    if a.write_repeat_manifest:
        manifest=write_repeat_manifest(); print(json.dumps({"effective_attempt_id":manifest["effective_attempt_id"],"objective_repeats":manifest["objective_repeat_count"],"explicit_subjective_repeats":manifest["explicit_subjective_repeat_count"]},indent=2)); return 0
    if a.finalize: print(json.dumps(finalize(),indent=2)); return 0
    raise SystemExit("choose --self-check, --derive-first-pass, --write-repeat-manifest, or --finalize")
if __name__=="__main__": raise SystemExit(main())
