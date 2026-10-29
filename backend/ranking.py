"""Deterministic ranking engine — no LLM involved."""
import re
from typing import List
from models import COA, RedTeamResult, RankedCOA


W_SUCCESS = 0.40
W_RISK = 0.30
W_LOGISTICS = 0.20
W_TIME = 0.10


def _risk_score(coa: COA, red_team) -> float:
    risk_count = len(coa.inherent_risks)
    if red_team:
        risk_count += len(red_team.vulnerabilities_exploited)
    capped = min(risk_count, 5)
    return 1.0 - (capped / 5.0)


def _logistics_confidence(coa: COA) -> float:
    plan_len = len(coa.logistics_plan.strip()) if coa.logistics_plan else 0
    asset_count = len(coa.required_assets)
    plan_score = min(plan_len / 200.0, 1.0)
    asset_score = min(asset_count / 3.0, 1.0)
    return round((plan_score * 0.6) + (asset_score * 0.4), 3)


def _time_score(coa: COA) -> float:
    text = (coa.estimated_time or "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.5
    value = float(match.group(1))
    if "day" in text:
        hours = value * 24
    elif "min" in text:
        hours = value / 60
    else:
        hours = value
    if hours <= 6:
        return 1.0
    if hours >= 72:
        return 0.0
    return round(1.0 - ((hours - 6) / (72 - 6)), 3)


def rank_coas(coas: List[COA], red_team_results: List[RedTeamResult]) -> List[RankedCOA]:
    """Rank all COAs deterministically. Returns sorted list (1 = best)."""
    if not coas:
        return []

    rt_map = {r.coa_id: r for r in red_team_results}
    ranked = []

    for coa in coas:
        rt = rt_map.get(coa.id)
        success_prob = rt.success_probability if rt else 0.5
        risk = _risk_score(coa, rt)
        logistics = _logistics_confidence(coa)
        time_s = _time_score(coa)

        score = round(
            W_SUCCESS * success_prob
            + W_RISK * risk
            + W_LOGISTICS * logistics
            + W_TIME * time_s,
            4,
        )

        breakdown = {
            "success_probability": {
                "raw": round(success_prob, 3),
                "weight": W_SUCCESS,
                "contribution": round(W_SUCCESS * success_prob, 4),
            },
            "risk_score": {
                "raw": round(risk, 3),
                "weight": W_RISK,
                "contribution": round(W_RISK * risk, 4),
            },
            "logistics_confidence": {
                "raw": round(logistics, 3),
                "weight": W_LOGISTICS,
                "contribution": round(W_LOGISTICS * logistics, 4),
            },
            "time_score": {
                "raw": round(time_s, 3),
                "weight": W_TIME,
                "contribution": round(W_TIME * time_s, 4),
            },
        }

        ranked.append({
            "coa_id": coa.id,
            "title": coa.title,
            "score": score,
            "score_breakdown": breakdown,
            "success_probability": round(success_prob, 3),
            "risk_score": round(risk, 3),
            "logistics_confidence": round(logistics, 3),
            "time_score": round(time_s, 3),
            "rank": 0,
            "explanation": None,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i

    return [RankedCOA(**item) for item in ranked]