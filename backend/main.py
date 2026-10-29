import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    Scenario, COA, RedTeamResult,
    ParseScenarioRequest, GenerateCOAsRequest, RedTeamRequest,
    RankCOAsRequest, ExplainRequest, ModifyCOARequest, ExportBriefRequest,
)
from llm_client import call_llm
from ranking import rank_coas
from prompts import (
    PARSE_SCENARIO_SYSTEM, parse_scenario_prompt,
    GENERATE_COAS_SYSTEM, generate_coas_prompt,
    RED_TEAM_SYSTEM, red_team_prompt,
    EXPLAIN_SYSTEM, explain_prompt,
    MODIFY_COA_SYSTEM, modify_coa_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WarMatrix Lite API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DECISION_LOG_PATH = DATA_DIR / "decision_log.json"


def _safe_json_loads(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}. Raw: {text[:300]}")
        raise HTTPException(500, f"LLM returned invalid JSON: {e}")


def _append_decision_log(entry: dict):
    try:
        log = json.loads(DECISION_LOG_PATH.read_text()) if DECISION_LOG_PATH.exists() else []
        entry["timestamp"] = datetime.now().isoformat()
        log.append(entry)
        DECISION_LOG_PATH.write_text(json.dumps(log, indent=2))
    except Exception as e:
        logger.warning(f"Decision log write failed: {e}")


@app.get("/")
def root():
    return {"status": "online", "service": "WarMatrix Lite", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/parse-scenario", response_model=Scenario)
def parse_scenario(req: ParseScenarioRequest):
    raw = call_llm(
        prompt=parse_scenario_prompt(req.raw_text),
        system_prompt=PARSE_SCENARIO_SYSTEM,
        temperature=0.2,
    )
    data = _safe_json_loads(raw)
    if data.get("error") == "LLM_UNAVAILABLE":
        raise HTTPException(503, "LLM unavailable.")
    return Scenario(**data)


@app.post("/generate-coas")
def generate_coas(req: GenerateCOAsRequest):
    raw = call_llm(
        prompt=generate_coas_prompt(req.scenario, req.count),
        system_prompt=GENERATE_COAS_SYSTEM,
        temperature=0.8,
    )
    data = _safe_json_loads(raw)
    if data.get("error") == "LLM_UNAVAILABLE":
        raise HTTPException(503, "LLM unavailable.")
    coas = [COA(**c) for c in data.get("coas", [])]
    return {"coas": coas, "count": len(coas)}


@app.post("/red-team", response_model=RedTeamResult)
def red_team(req: RedTeamRequest):
    raw = call_llm(
        prompt=red_team_prompt(req.coa, req.scenario, req.persona),
        system_prompt=RED_TEAM_SYSTEM,
        temperature=0.6,
    )
    data = _safe_json_loads(raw)
    if data.get("error") == "LLM_UNAVAILABLE":
        raise HTTPException(503, "LLM unavailable.")
    prob = float(data.get("success_probability", 0.5))
    data["success_probability"] = max(0.0, min(1.0, prob))
    return RedTeamResult(**data)


@app.post("/rank-coas")
def rank_coas_endpoint(req: RankCOAsRequest):
    ranked = rank_coas(req.coas, req.red_team_results)
    return {"ranked_coas": ranked}


@app.post("/explain")
def explain(req: ExplainRequest):
    raw = call_llm(
        prompt=explain_prompt(req.coa, req.ranked, req.scenario),
        system_prompt=EXPLAIN_SYSTEM,
        temperature=0.4,
    )
    data = _safe_json_loads(raw)
    if data.get("error") == "LLM_UNAVAILABLE":
        return {"explanation": "Explanation temporarily unavailable."}
    return {"explanation": data.get("explanation", "")}


@app.post("/modify-coa", response_model=COA)
def modify_coa(req: ModifyCOARequest):
    raw = call_llm(
        prompt=modify_coa_prompt(req.coa, req.instruction),
        system_prompt=MODIFY_COA_SYSTEM,
        temperature=0.6,
    )
    data = _safe_json_loads(raw)
    if data.get("error") == "LLM_UNAVAILABLE":
        raise HTTPException(503, "LLM unavailable.")
    modified = COA(**data)
    _append_decision_log({
        "action": "modify_coa",
        "coa_id": req.coa.id,
        "instruction": req.instruction,
        "before_title": req.coa.title,
        "after_title": modified.title,
    })
    return modified


@app.get("/decision-log")
def get_decision_log():
    if not DECISION_LOG_PATH.exists():
        return {"log": []}
    try:
        return {"log": json.loads(DECISION_LOG_PATH.read_text())}
    except Exception:
        return {"log": []}


@app.post("/export-brief")
def export_brief(req: ExportBriefRequest):
    lines = [
        "# WarMatrix Lite — Decision Brief",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## Scenario\n",
        f"**Objective:** {req.scenario.mission_objective}",
        f"**Terrain:** {req.scenario.terrain}",
        f"**Weather:** {req.scenario.weather}",
        f"**Enemy:** {req.scenario.enemy_strength}",
        f"**Time constraint:** {req.scenario.time_constraint}",
    ]
    if req.scenario.intelligence_gaps:
        lines.append(f"**Intelligence gaps:** {', '.join(req.scenario.intelligence_gaps)}")
    lines.append("\n## Ranked Courses of Action\n")
    lines.append("| Rank | COA | Score | Success | Risk | Logistics | Time |")
    lines.append("|------|-----|-------|---------|------|-----------|------|")
    for r in req.ranked_coas:
        lines.append(
            f"| {r.rank} | {r.title} | {r.score:.3f} | {r.success_probability:.2f} | "
            f"{r.risk_score:.2f} | {r.logistics_confidence:.2f} | {r.time_score:.2f} |"
        )
    lines.append("\n## Top Recommendation\n")
    if req.ranked_coas:
        top = req.ranked_coas[0]
        lines.append(f"**{top.title}** — Score: {top.score:.3f}\n")
        if top.explanation:
            lines.append(top.explanation)
    lines.append("\n## Red Team Notes\n")
    for rt in req.red_team_results:
        lines.append(f"### {rt.coa_id} vs {rt.persona}")
        lines.append(f"**Success Probability:** {rt.success_probability:.2f} ({rt.confidence} confidence)")
        lines.append("**Adversary actions:**")
        for a in rt.adversary_actions:
            lines.append(f"- {a}")
        lines.append("**Vulnerabilities exploited:**")
        for v in rt.vulnerabilities_exploited:
            lines.append(f"- {v}")
        lines.append("")
    lines.append("\n---\n*Decision support only. Final call rests with the human commander.*")
    return {"markdown": "\n".join(lines)}


SCENARIOS_FILE = DATA_DIR / "scenarios.json"


@app.get("/scenarios")
def get_scenarios():
    if not SCENARIOS_FILE.exists():
        return {"scenarios": []}
    try:
        return {"scenarios": json.loads(SCENARIOS_FILE.read_text())}
    except Exception:
        return {"scenarios": []}