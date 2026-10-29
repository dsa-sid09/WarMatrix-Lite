"""All LLM prompts live here."""
from models import Scenario, COA, RedTeamResult, RankedCOA


# ============================================================
# 1. PARSE SCENARIO
# ============================================================

PARSE_SCENARIO_SYSTEM = (
    "You are a military intelligence analyst. "
    "Convert free-text mission briefings into structured JSON. "
    "Respond ONLY with valid JSON matching the requested schema. No prose."
)

PARSE_SCENARIO_TEMPLATE = """Parse the following mission briefing into a structured scenario.

Return JSON with EXACTLY these keys:
{{
  "mission_objective": "string — the primary goal",
  "terrain": "string — terrain type",
  "weather": "string — weather conditions",
  "enemy_strength": "string — enemy size/type",
  "friendly_assets": ["list", "of", "friendly", "units"],
  "time_constraint": "string — time available",
  "intelligence_gaps": ["list", "of", "unknowns"]
}}

If a field is missing, infer a reasonable military default.

MISSION BRIEFING:
{raw_text}
"""


def parse_scenario_prompt(raw_text: str) -> str:
    return PARSE_SCENARIO_TEMPLATE.format(raw_text=raw_text)


# ============================================================
# 2. GENERATE COAs
# ============================================================

GENERATE_COAS_SYSTEM = (
    "You are a senior military operations planner. "
    "Generate distinct, tactically sound Courses of Action (COAs). "
    "Respond ONLY with valid JSON. No prose, no markdown."
)

GENERATE_COAS_TEMPLATE = """Given this scenario, generate exactly {count} DISTINCT COAs.

SCENARIO:
{scenario_json}

Each COA must be meaningfully DIFFERENT in doctrine, tempo, axis, or force mix.

Return JSON:
{{
  "coas": [
    {{
      "id": "COA-1",
      "title": "Short 3-5 word name",
      "summary": "Two sentence summary.",
      "steps": ["Step 1", "Step 2", "Step 3"],
      "required_assets": ["asset 1", "asset 2"],
      "estimated_time": "18 hours",
      "logistics_plan": "Detailed sentence about fuel, ammo, medical, resupply.",
      "inherent_risks": ["risk 1", "risk 2"]
    }}
  ]
}}

RULES:
- ids MUST be "COA-1" through "COA-{count}"
- steps: 3-6 sequential actions
- required_assets: 2-5 items
- logistics_plan: at least one full sentence mentioning fuel/ammo/medical
- inherent_risks: 2-5 items
"""


def generate_coas_prompt(scenario: Scenario, count: int = 5) -> str:
    return GENERATE_COAS_TEMPLATE.format(
        count=count,
        scenario_json=scenario.model_dump_json(indent=2),
    )


# ============================================================
# 3. RED TEAM
# ============================================================

PERSONA_BLURBS = {
    "Aggressive": (
        "You are an AGGRESSIVE enemy commander. Attack hard, fast, early. "
        "Exploit tempo and shock. Accept higher casualties for decisive results."
    ),
    "Cautious": (
        "You are a CAUTIOUS enemy commander. Trade space for time. "
        "Fortify, ambush, preserve force. Wait for the attacker to overextend."
    ),
    "Supply-Line Cutter": (
        "You are a SUPPLY-LINE CUTTER. Avoid frontal combat. "
        "Strike convoys, bridges, and command nodes. Starve the attacker of sustainment."
    ),
}

RED_TEAM_SYSTEM = (
    "You are a Red Team adversary simulating enemy reactions to a friendly COA. "
    "Respond ONLY with valid JSON. Be specific and realistic."
)

RED_TEAM_TEMPLATE = """{persona_blurb}

FRIENDLY SCENARIO:
{scenario_json}

FRIENDLY COA BEING STRESS-TESTED:
{coa_json}

Analyze how you respond as this adversary. Identify specific vulnerabilities.

Return JSON:
{{
  "coa_id": "{coa_id}",
  "persona": "{persona}",
  "adversary_actions": ["action 1", "action 2", "action 3"],
  "vulnerabilities_exploited": ["vuln 1", "vuln 2", "vuln 3"],
  "success_probability": 0.65,
  "confidence": "High"
}}

RULES:
- success_probability: float 0.0 to 1.0 (friendly COA's chance of success)
- More exploited vulnerabilities = LOWER success_probability
- confidence: "High", "Medium", or "Low"
"""


def red_team_prompt(coa: COA, scenario: Scenario, persona: str = "Aggressive") -> str:
    return RED_TEAM_TEMPLATE.format(
        persona_blurb=PERSONA_BLURBS.get(persona, PERSONA_BLURBS["Aggressive"]),
        scenario_json=scenario.model_dump_json(indent=2),
        coa_json=coa.model_dump_json(indent=2),
        coa_id=coa.id,
        persona=persona,
    )


# ============================================================
# 4. EXPLAIN
# ============================================================

EXPLAIN_SYSTEM = (
    "You are a military staff officer explaining a wargaming recommendation. "
    "Be concise, specific, honest about weaknesses. Respond ONLY with valid JSON."
)

EXPLAIN_TEMPLATE = """Explain WHY this COA ranked where it did.

SCENARIO:
{scenario_json}

COA:
{coa_json}

DETERMINISTIC RANKING (already computed — do NOT recalculate):
{ranked_json}

Write 3 sentences covering:
1. Why this COA scored where it did (reference score components)
2. Main risk or intelligence gap
3. One concrete action to improve the score

Return JSON:
{{
  "explanation": "Sentence one. Sentence two. Sentence three."
}}
"""


def explain_prompt(coa: COA, ranked: RankedCOA, scenario: Scenario) -> str:
    return EXPLAIN_TEMPLATE.format(
        scenario_json=scenario.model_dump_json(indent=2),
        coa_json=coa.model_dump_json(indent=2),
        ranked_json=ranked.model_dump_json(indent=2),
    )


# ============================================================
# 5. MODIFY COA
# ============================================================

MODIFY_COA_SYSTEM = (
    "You are a military planner modifying a COA per the commander's instruction. "
    "Keep the same COA id. Return ONLY valid JSON in the same schema."
)

MODIFY_COA_TEMPLATE = """MODIFY the following COA per the commander's instruction.

CURRENT COA:
{coa_json}

COMMANDER'S INSTRUCTION:
{instruction}

Apply the instruction while keeping the COA tactically coherent.
Do NOT change the COA id. Update all affected fields.

Return JSON with same schema as the input COA:
{{
  "id": "{coa_id}",
  "title": "string",
  "summary": "string",
  "steps": ["..."],
  "required_assets": ["..."],
  "estimated_time": "string",
  "logistics_plan": "string",
  "inherent_risks": ["..."]
}}
"""


def modify_coa_prompt(coa: COA, instruction: str) -> str:
    return MODIFY_COA_TEMPLATE.format(
        coa_json=coa.model_dump_json(indent=2),
        instruction=instruction,
        coa_id=coa.id,
    )