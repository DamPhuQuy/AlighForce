#!/usr/bin/env python3
"""Offline SACH utilities. No model calls, command execution, or implicit file writes.

Validation supports only the schema vocabulary used by the bundled contracts;
it is not a general JSON Schema implementation or a formal program verifier.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schema_errors(value, schema, root=None, location="$"):
    root = schema if root is None else root
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/"):
            return [f"{location}: unsupported schema reference"]
        return schema_errors(value, root["$defs"][ref.split("/")[-1]], root, location)
    errors = []
    kind = schema.get("type")
    valid = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
             "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
             "integer": lambda v: isinstance(v, int) and not isinstance(v, bool)}
    if kind and (kind not in valid or not valid[kind](value)):
        return [f"{location}: expected {kind}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location}: invalid enum value")
    if isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0): errors.append(f"{location}: empty/short string")
        if len(value) > schema.get("maxLength", len(value)): errors.append(f"{location}: string too long")
        if "pattern" in schema and not re.search(schema["pattern"], value): errors.append(f"{location}: pattern mismatch")
    if isinstance(value, int) and not isinstance(value, bool):
        if value < schema.get("minimum", value): errors.append(f"{location}: below minimum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value: errors.append(f"{location}: missing {key}")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                errors += schema_errors(item, schema["properties"][key], root, f"{location}.{key}")
            elif schema.get("additionalProperties") is False: errors.append(f"{location}: unknown {key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0): errors.append(f"{location}: too few items")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            errors.append(f"{location}: duplicate items")
        for i, item in enumerate(value):
            errors += schema_errors(item, schema.get("items", {}), root, f"{location}[{i}]")
    return errors


def validate_contracts(tasks, specs):
    if not isinstance(tasks, list) or not isinstance(specs, list):
        return ["tasks and specs must be lists"]
    base = Path(__file__).parent / "schemas"
    errors = []
    for values, name in [(specs, "spec"), (tasks, "task")]:
        schema = read_json(base / f"{name}.schema.json")
        for i, value in enumerate(values): errors += schema_errors(value, schema, location=f"{name}[{i}]")
    if errors: return errors
    spec_map = {s["spec_id"]: s for s in specs}
    task_map = {t["task_id"]: t for t in tasks}
    if len(spec_map) != len(specs): errors.append("duplicate spec_id")
    if len(task_map) != len(tasks): errors.append("duplicate task_id")
    for spec in specs:
        req_ids = [r["id"] for r in spec["ears_requirements"]]
        scenario_ids = [s["id"] for s in spec["bdd_scenarios"]]
        if len(set(req_ids)) != len(req_ids) or len(set(scenario_ids)) != len(scenario_ids):
            errors.append("duplicate requirement/scenario ID")
        covered = set()
        for req in spec["ears_requirements"]:
            pattern = req["pattern"]
            if pattern in ("event_driven", "unwanted_behavior", "complex") and not req.get("trigger", "").strip():
                errors.append(f"{req['id']}: trigger required")
            if pattern in ("state_driven", "optional_feature") and not req.get("precondition", "").strip():
                errors.append(f"{req['id']}: precondition required")
            if "THE SYSTEM SHALL " not in req["full_statement"].upper():
                errors.append(f"{req['id']}: missing THE SYSTEM SHALL response")
        for scenario in spec["bdd_scenarios"]:
            for ref in scenario["requirement_ids"]:
                if ref not in req_ids: errors.append(f"{scenario['id']}: unknown requirement {ref}")
                covered.add(ref)
        for ref in set(req_ids) - covered: errors.append(f"{ref}: no acceptance scenario")
    for task in tasks:
        if task["spec_ref"] not in spec_map: errors.append(f"{task['task_id']}: unknown spec_ref")
        for path in task["allowed_files"]:
            normalized = path.replace("\\", "/")
            if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in normalized.split("/") or normalized in (".", "**") or normalized.split("/")[0] == ".git":
                errors.append(f"{task['task_id']}: unsafe allowed path {path}")
        for dep in task["dependencies"]:
            if dep not in task_map: errors.append(f"{task['task_id']}: missing dependency {dep}")
        if task.get("execution_frontier") and (task["status"] not in ("PENDING", "READY") or any(task_map.get(d, {}).get("status") != "COMPLETED" for d in task["dependencies"])):
            errors.append(f"{task['task_id']}: invalid execution frontier")
        if task["status"] in ("READY", "IN_PROGRESS", "COMPLETED") and any(task_map.get(d, {}).get("status") != "COMPLETED" for d in task["dependencies"]):
            errors.append(f"{task['task_id']}: active/completed task has unfinished dependencies")
        goals = task["binary_goals"]
        if len({g["id"] for g in goals}) != len(goals): errors.append("duplicate goal ID")
        if task["status"] == "COMPLETED" and (not all(g["passed"] for g in goals) or not task["verifier"].get("evidence_ref")):
            errors.append(f"{task['task_id']}: completion requires passing goals and evidence_ref")
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting: errors.append(f"dependency cycle at {key}"); return
        if key in visited: return
        visiting.add(key)
        for dep in task_map[key]["dependencies"]:
            if dep in task_map: visit(dep)
        visiting.remove(key); visited.add(key)
    for key in task_map: visit(key)
    return errors


def assemble_context(pack, budget, reserve):
    """Budget the content using UTF-8 bytes as a conservative token upper bound.

    Actual provider framing/tool schemas are outside this helper; reserve for them.
    The static block is byte-preserved and optional observations are omitted whole.
    """
    if budget <= 0 or reserve < 0 or reserve >= budget: raise ValueError("invalid context budget/reserve")
    if not isinstance(pack, dict): raise ValueError("context input must be an object")
    static = pack["static"]
    if not isinstance(static, str): raise ValueError("static must be a string")
    entries = pack.get("dynamic", [])
    if not isinstance(entries, list): raise ValueError("dynamic must be an array")
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict): raise ValueError("context entry must be an object")
        if not isinstance(entry.get("text"), str) or not isinstance(entry.get("id"), str): raise ValueError("dynamic entries require id and text")
        if not isinstance(entry.get("required", False), bool): raise ValueError("required must be boolean")
        if not isinstance(entry.get("priority", 0), int) or isinstance(entry.get("priority", 0), bool): raise ValueError("priority must be integer")
        if entry["id"] in ids: raise ValueError("duplicate context entry ID")
        ids.add(entry["id"])
    cost = lambda text: len(text.encode("utf-8"))
    capacity = budget - reserve
    required = [e for e in entries if e.get("required", False)]
    used = cost(static) + sum(cost("\n" + e["text"]) for e in required)
    if used > capacity: raise ValueError("required context exceeds budget; checkpoint or enlarge budget")
    chosen = {e["id"] for e in required}
    for entry in sorted(entries, key=lambda e: -e.get("priority", 0)):
        if entry["id"] in chosen: continue
        size = cost("\n" + entry["text"])
        if used + size <= capacity: chosen.add(entry["id"]); used += size
    text = static + "".join("\n" + e["text"] for e in entries if e["id"] in chosen)
    return {"text": text, "static_sha256": digest(static.encode("utf-8")),
            "content_token_upper_bound": used, "reserve_tokens": reserve,
            "estimator": "utf8-bytes-upper-bound; excludes provider framing",
            "included": [e["id"] for e in entries if e["id"] in chosen],
            "omitted": [{"id": e["id"], "source": e.get("source"), "sha256": digest(e["text"].encode("utf-8"))} for e in entries if e["id"] not in chosen]}


def check_memory(entries, root):
    if not isinstance(entries, list): raise ValueError("memory must be a list")
    root = Path(root).resolve()
    results = []
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip(): raise ValueError("memory requires a nonempty ID")
        if entry["id"] in ids: raise ValueError("duplicate memory ID")
        ids.add(entry["id"])
        state = "unverified"
        if entry.get("expires_at"):
            if not isinstance(entry["expires_at"], str): raise ValueError("expiry must be an ISO timestamp")
            expiry = datetime.fromisoformat(entry["expires_at"].replace("Z", "+00:00"))
            if expiry.tzinfo is None: raise ValueError("memory expiry requires a timezone")
            if expiry <= datetime.now(timezone.utc): state = "expired"
        sources = entry.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(s, dict) or not isinstance(s.get("path"), str) for s in sources): raise ValueError("memory sources require local paths")
        if state != "expired" and sources:
            state = "valid"
            for source in sources:
                target = (root / source["path"]).resolve()
                if not target.is_relative_to(root): raise ValueError("memory source outside repository")
                if not target.is_file() or digest(target.read_bytes()) != source.get("sha256"): state = "stale"
        results.append({"id": entry["id"], "status": state})
    return results


def graph_query(graph, terms, hops=1, max_nodes=20, max_chars=12000):
    if hops not in (0, 1, 2) or not 1 <= max_nodes <= 50: raise ValueError("graph limits: hops 0..2; nodes 1..50")
    if not 1024 <= max_chars <= 64000: raise ValueError("graph output budget: 1024..64000 characters")
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list): raise ValueError("graph requires a nodes array")
    if any(not isinstance(n, dict) or "id" not in n for n in graph["nodes"]): raise ValueError("graph nodes require IDs")
    nodes = {str(n["id"]): n for n in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]): raise ValueError("duplicate graph node ID")
    scored = sorted(((sum(t.casefold() in str(n.get("label", key)).casefold() for t in terms), key) for key, n in nodes.items()), key=lambda x: (-x[0], x[1]))
    seeds = [key for score, key in scored if score][:min(3, max_nodes)]
    adjacency = {key: set() for key in nodes}
    edges = graph.get("links", graph.get("edges", []))
    if not isinstance(edges, list) or any(not isinstance(e, dict) or "source" not in e or "target" not in e for e in edges): raise ValueError("graph edges require source and target")
    for edge in edges:
        a, b = str(edge["source"]), str(edge["target"])
        if a in nodes and b in nodes: adjacency[a].add(b); adjacency[b].add(a)
    selected, frontier = set(seeds), seeds
    for _ in range(hops):
        next_frontier = []
        for key in frontier:
            for neighbor in sorted(adjacency[key]):
                if neighbor not in selected and len(selected) < max_nodes:
                    selected.add(neighbor); next_frontier.append(neighbor)
        frontier = next_frontier
    # Only return source metadata and bounded labels, never raw graph-embedded instructions/code.
    ordered = seeds + sorted(selected - set(seeds))
    result = {"seeds": seeds, "freshness": "unverified; check source locations against current checkout",
              "truncated": False, "ignored_dangling_edges": sum(str(e["source"]) not in nodes or str(e["target"]) not in nodes for e in edges),
              "nodes": [{k: str(v)[:500] for k, v in nodes[key].items() if k in ("id", "label", "source_file", "source_location")} for key in ordered],
              "edges": [{k: str(v)[:500] for k, v in e.items() if k in ("source", "target", "relation", "confidence")} for e in edges if str(e["source"]) in selected and str(e["target"]) in selected][:max_nodes * 4]}
    while len(json.dumps(result, ensure_ascii=True, indent=2)) > max_chars:
        result["truncated"] = True
        if result["edges"]:
            result["edges"].pop()
        elif result["nodes"]:
            removed = result["nodes"].pop()["id"]
            result["seeds"] = [s for s in result["seeds"] if s != removed]
        elif result["seeds"]:
            result["seeds"].pop()
        else: break
    return result


def generate_research_queries(topic, domain="software", target="modern_tech"):
    """Generate structured search queries to explore primary docs, benchmarks, modern tech, and CVE/failure modes."""
    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("research query requires a non-empty topic")
    queries = {
        "primary_specs_and_docs": [
            f"{clean_topic} official documentation specification RFC",
            f"{clean_topic} API reference architecture standards"
        ],
        "proven_benchmarks_and_cases": [
            f"{clean_topic} production benchmark performance latency case study",
            f"{clean_topic} battle-tested architecture scale high-throughput"
        ],
        "modern_tech_and_alternatives": [
            f"{clean_topic} modern alternatives state-of-the-art libraries 2025 2026",
            f"{clean_topic} zero-dependency lightweight modern replacement"
        ],
        "adversarial_edge_cases_and_cves": [
            f"{clean_topic} common pitfalls failure modes memory leak race condition",
            f"{clean_topic} security vulnerabilities CVE postmortem edge cases"
        ]
    }
    return {
        "topic": clean_topic,
        "domain": domain,
        "target": target,
        "query_count": sum(len(v) for v in queries.values()),
        "query_sets": queries
    }


URL_PATTERN = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
DOI_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", re.IGNORECASE)


def is_valid_url_or_doi(val):
    if not isinstance(val, str) or not val.strip():
        return False
    val = val.strip()
    return bool(URL_PATTERN.match(val) or DOI_PATTERN.match(val))


def validate_research_dossier(dossier, strict=False):
    """Validate research dossier with support for contract v2 and legacy compatibility."""
    if not isinstance(dossier, dict):
        raise ValueError("research dossier must be an object")
    task_id = dossier.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("research dossier requires a non-empty task_id")

    schema_version = dossier.get("schema_version")
    if schema_version == "2.0":
        base = Path(__file__).parent / "schemas"
        schema = read_json(base / "research-dossier.schema.json")
        errors = schema_errors(dossier, schema)
        warnings = []
        sources = dossier.get("sources", [])
        source_ids = set()
        for i, src in enumerate(sources):
            if not isinstance(src, dict):
                continue
            sid = src.get("source_id")
            if sid in source_ids:
                errors.append(f"sources[{i}]: duplicate source_id '{sid}'")
            elif sid:
                source_ids.add(sid)
            url_or_doi = src.get("url_or_doi", "")
            if not is_valid_url_or_doi(url_or_doi):
                errors.append(f"sources[{i}]: invalid URL or DOI format: {url_or_doi}")
            if "arxiv.org" in str(url_or_doi).lower() and src.get("publication_status") == "peer_reviewed":
                warnings.append(f"sources[{i}]: arXiv preprints are unrefereed; verify peer review venue in limitations")

        claims = dossier.get("claims", [])
        claim_ids = set()
        for j, clm in enumerate(claims):
            if not isinstance(clm, dict):
                continue
            cid = clm.get("claim_id")
            if cid in claim_ids:
                errors.append(f"claims[{j}]: duplicate claim_id '{cid}'")
            elif cid:
                claim_ids.add(cid)
            for s_ref in clm.get("source_ids", []):
                if s_ref not in source_ids:
                    errors.append(f"claims[{j}]: unknown source reference '{s_ref}'")
            if clm.get("epistemic_status") == "unverified":
                warnings.append(f"claims[{j}]: claim '{cid}' is unverified")

        readiness = dossier.get("execution_readiness")
        if isinstance(readiness, dict):
            for ref in readiness.get("evidence_refs", []):
                if ref not in claim_ids and ref not in source_ids:
                    errors.append(f"execution_readiness: unknown evidence reference '{ref}'")

        if strict:
            if not claims:
                errors.append("strict mode: research dossier requires at least one claim")
            if not isinstance(readiness, dict) or not readiness.get("ready"):
                errors.append("strict mode: execution_readiness.ready must be true")
            if isinstance(readiness, dict) and not readiness.get("evidence_refs"):
                errors.append("strict mode: execution_readiness requires non-empty evidence_refs")

        return {
            "valid": not errors,
            "schema_version": "2.0",
            "validation_scope": "contract_v2_structural",
            "strict": strict,
            "errors": errors,
            "warnings": warnings,
            "task_id": task_id,
            "total_sources": len(sources),
            "total_claims": len(claims),
            "execution_readiness": readiness
        }

    # Legacy path (v1 / unversioned)
    if strict:
        return {
            "valid": False,
            "schema_version": "legacy_v1",
            "validation_scope": "legacy_rejected_in_strict",
            "strict": True,
            "errors": ["strict mode requires dossier v2 with schema_version: '2.0'"],
            "warnings": [],
            "task_id": task_id
        }

    sources = dossier.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("research dossier requires at least one source")

    valid_tiers = {"Tier-A", "Tier-B", "Tier-C"}
    valid_relevance = {"high", "medium", "low"}
    errors = []
    warnings = [
        "Legacy dossier format: caller-declared tiers are metadata only and do not establish verification.",
        "evidence_confidence_index is deprecated and based solely on unverified caller claims."
    ]
    tier_counts = {"Tier-A": 0, "Tier-B": 0, "Tier-C": 0}

    for i, src in enumerate(sources):
        loc = f"sources[{i}]"
        if not isinstance(src, dict):
            errors.append(f"{loc}: must be an object")
            continue
        title = src.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"{loc}: title must be a non-empty string")
        url_or_doi = src.get("url_or_doi")
        if not isinstance(url_or_doi, str) or not url_or_doi.strip():
            errors.append(f"{loc}: url_or_doi must be a non-empty string")
        elif not is_valid_url_or_doi(url_or_doi):
            warnings.append(f"{loc}: url_or_doi '{url_or_doi}' is not a valid URL or DOI syntax")
        tier = src.get("tier")
        if tier not in valid_tiers:
            errors.append(f"{loc}: invalid tier {tier}; expected Tier-A, Tier-B, or Tier-C")
        else:
            tier_counts[tier] += 1
        rel = src.get("relevance", "medium")
        if rel not in valid_relevance:
            errors.append(f"{loc}: invalid relevance {rel}; expected high, medium, or low")
        findings = src.get("key_findings")
        if not isinstance(findings, str) or not findings.strip():
            errors.append(f"{loc}: key_findings must be a non-empty string")

    tech_eval = dossier.get("technologies_evaluated", [])
    if not isinstance(tech_eval, list):
        errors.append("technologies_evaluated must be a list")
    else:
        valid_statuses = {"adopted", "rejected", "investigating"}
        for i, tech in enumerate(tech_eval):
            loc = f"technologies_evaluated[{i}]"
            if not isinstance(tech, dict):
                errors.append(f"{loc}: must be an object")
                continue
            name = tech.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{loc}: name must be a non-empty string")
            status = tech.get("status")
            if status not in valid_statuses:
                errors.append(f"{loc}: status must be adopted, rejected, or investigating")
            tradeoffs = tech.get("tradeoffs")
            if not isinstance(tradeoffs, (str, list)):
                errors.append(f"{loc}: tradeoffs must be specified")

    if (tier_counts["Tier-A"] + tier_counts["Tier-B"]) == 0:
        errors.append("research credibility requires at least one Tier-A or Tier-B source")

    total_sources = len(sources)
    eci = round((tier_counts["Tier-A"] * 1.0 + tier_counts["Tier-B"] * 0.7 + tier_counts["Tier-C"] * 0.3) / max(1, total_sources), 3)

    return {
        "valid": not errors,
        "schema_version": "legacy_v1",
        "validation_scope": "legacy_structure_only",
        "strict": False,
        "errors": errors,
        "warnings": warnings,
        "task_id": task_id,
        "total_sources": total_sources,
        "tier_distribution": tier_counts,
        "evidence_confidence_index": eci,
        "evidence_confidence_index_deprecated": True,
        "technologies_evaluated_count": len(tech_eval) if isinstance(tech_eval, list) else 0
    }


def validate_debate_ledger(ledger, strict=False):
    """Validate multi-expert adversarial review ledger with v2 and legacy support."""
    if not isinstance(ledger, dict):
        raise ValueError("debate ledger must be an object")
    task_id = ledger.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("debate ledger requires a non-empty task_id")

    schema_version = ledger.get("schema_version")
    if schema_version == "2.0":
        base = Path(__file__).parent / "schemas"
        schema = read_json(base / "debate-ledger.schema.json")
        errors = schema_errors(ledger, schema)
        warnings = []
        tier = ledger.get("complexity_tier", 0)
        experts = ledger.get("expert_panel", [])
        seen_roles = set()
        for i, exp in enumerate(experts):
            if not isinstance(exp, dict):
                continue
            role = exp.get("role")
            if role in seen_roles:
                errors.append(f"expert_panel[{i}]: duplicate expert role '{role}'")
            elif role:
                seen_roles.add(role)
            challenges = exp.get("challenges") or exp.get("attacks") or []
            if tier >= 1 and not challenges:
                errors.append(f"expert_panel[{i}]: tier {tier} requires non-empty challenges/counterexamples")
            for j, ch in enumerate(challenges):
                if not isinstance(ch, dict) or not ch:
                    errors.append(f"expert_panel[{i}].challenges[{j}]: empty challenge object")
                elif not ch.get("counterexample", "").strip():
                    errors.append(f"expert_panel[{i}].challenges[{j}]: counterexample must be non-empty")

        tradeoffs = ledger.get("tradeoffs_matrix", [])
        approaches = {t.get("approach") for t in tradeoffs if isinstance(t, dict) and t.get("approach")}
        pareto = ledger.get("pareto_selection")
        if tier >= 1 and pareto:
            chosen = pareto.get("chosen_approach")
            if approaches and chosen not in approaches:
                errors.append(f"pareto_selection: chosen_approach '{chosen}' not found in tradeoffs_matrix")

        if strict and tier >= 1 and not tradeoffs:
            errors.append("strict mode: tier >= 1 requires non-empty tradeoffs_matrix")

        return {
            "valid": not errors,
            "schema_version": "2.0",
            "validation_scope": "contract_v2_structural",
            "strict": strict,
            "errors": errors,
            "warnings": warnings,
            "task_id": task_id,
            "complexity_tier": tier,
            "expert_count": len(experts),
            "approaches_evaluated": list(approaches)
        }

    # Legacy path (v1 / unversioned)
    if strict:
        return {
            "valid": False,
            "schema_version": "legacy_v1",
            "validation_scope": "legacy_rejected_in_strict",
            "strict": True,
            "errors": ["strict mode requires debate ledger v2 with schema_version: '2.0'"],
            "warnings": [],
            "task_id": task_id
        }

    tier = ledger.get("complexity_tier")
    if not isinstance(tier, int) or isinstance(tier, bool) or tier not in (0, 1, 2, 3):
        raise ValueError("complexity_tier must be integer in 0..3")

    min_experts = {0: 0, 1: 1, 2: 3, 3: 4}[tier]
    experts = ledger.get("expert_panel", [])
    if not isinstance(experts, list):
        raise ValueError("expert_panel must be a list")

    errors = []
    warnings = [
        "Legacy debate ledger format: expert count alone does not establish empirical verification.",
        "Legacy attacks must be evaluated with concrete, executable counterexamples."
    ]
    if len(experts) < min_experts:
        errors.append(f"complexity tier {tier} requires at least {min_experts} expert persona(s), found {len(experts)}")

    seen_roles = set()
    for i, exp in enumerate(experts):
        loc = f"expert_panel[{i}]"
        if not isinstance(exp, dict):
            errors.append(f"{loc}: must be an object")
            continue
        role = exp.get("role")
        if not isinstance(role, str) or not role.strip():
            errors.append(f"{loc}: role must be non-empty")
        elif role in seen_roles:
            errors.append(f"{loc}: duplicate expert role '{role}'")
        else:
            seen_roles.add(role)

        attacks = exp.get("attacks", [])
        if not isinstance(attacks, list):
            errors.append(f"{loc}: attacks must be a list")
        elif tier >= 1 and not attacks:
            errors.append(f"{loc}: tier {tier} requires non-empty attacks/counterexamples")
        else:
            for j, att in enumerate(attacks):
                if isinstance(att, dict) and not att:
                    errors.append(f"{loc}.attacks[{j}]: empty attack object without counterexample substance")
                elif isinstance(att, str):
                    warnings.append(f"{loc}.attacks[{j}]: legacy string attack is unverified as executable counterexample")

    tradeoffs = ledger.get("tradeoffs_matrix", [])
    if not isinstance(tradeoffs, list):
        errors.append("tradeoffs_matrix must be a list")
    elif tier >= 1 and not tradeoffs:
        errors.append(f"complexity tier {tier} requires a non-empty tradeoffs_matrix")

    approaches = set()
    if isinstance(tradeoffs, list):
        for i, t in enumerate(tradeoffs):
            loc = f"tradeoffs_matrix[{i}]"
            if not isinstance(t, dict) or not isinstance(t.get("approach"), str):
                errors.append(f"{loc}: requires an approach string")
            else:
                approaches.add(t["approach"])

    pareto = ledger.get("pareto_selection")
    if tier >= 1:
        if not isinstance(pareto, dict):
            errors.append("pareto_selection must be an object with chosen_approach, justification, and mitigated_risks")
        else:
            chosen = pareto.get("chosen_approach")
            if not isinstance(chosen, str) or not chosen.strip():
                errors.append("pareto_selection: chosen_approach must be non-empty")
            elif approaches and chosen not in approaches:
                errors.append(f"pareto_selection: chosen_approach '{chosen}' not found in tradeoffs_matrix")
            if not isinstance(pareto.get("justification"), str) or not pareto.get("justification").strip():
                errors.append("pareto_selection: justification must be non-empty")

    return {
        "valid": not errors,
        "schema_version": "legacy_v1",
        "validation_scope": "legacy_structure_only",
        "strict": False,
        "errors": errors,
        "warnings": warnings,
        "task_id": task_id,
        "complexity_tier": tier,
        "expert_count": len(experts),
        "approaches_evaluated": list(approaches)
    }


def validate_capability_catalog(catalog):
    """Validate capability catalog against the capability-catalog schema."""
    if not isinstance(catalog, dict):
        raise ValueError("capability catalog must be an object")
    base = Path(__file__).parent / "schemas"
    schema = read_json(base / "capability-catalog.schema.json")
    errors = schema_errors(catalog, schema)
    caps = catalog.get("capabilities", [])
    seen_ids = set()
    for i, cap in enumerate(caps):
        if not isinstance(cap, dict):
            continue
        cid = cap.get("capability_id")
        if cid in seen_ids:
            errors.append(f"capabilities[{i}]: duplicate capability_id '{cid}'")
        elif cid:
            seen_ids.add(cid)
    return {
        "valid": not errors,
        "errors": errors,
        "capability_count": len(caps)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--task", action="append", required=True)
    validate.add_argument("--spec", action="append", required=True)
    context = sub.add_parser("context")
    context.add_argument("--input", required=True)
    context.add_argument("--budget-tokens", type=int, required=True)
    context.add_argument("--reserve-tokens", type=int, default=2048)
    memory = sub.add_parser("memory-check")
    memory.add_argument("--input", required=True)
    memory.add_argument("--root", default=".")
    graph = sub.add_parser("graph-query")
    graph.add_argument("--input", required=True)
    graph.add_argument("--term", action="append", required=True)
    graph.add_argument("--hops", type=int, default=1)
    graph.add_argument("--max-nodes", type=int, default=20)
    graph.add_argument("--max-chars", type=int, default=12000)
    r_queries = sub.add_parser("research-queries")
    r_queries.add_argument("--topic", required=True)
    r_queries.add_argument("--domain", default="software")
    r_queries.add_argument("--target", default="modern_tech")
    r_dossier = sub.add_parser("research-dossier")
    r_dossier.add_argument("--input", required=True)
    r_dossier.add_argument("--strict", action="store_true", help="Enforce contract v2 and completeness")
    d_eval = sub.add_parser("debate-eval")
    d_eval.add_argument("--input", required=True)
    d_eval.add_argument("--strict", action="store_true", help="Enforce contract v2 and completeness")
    c_check = sub.add_parser("capability-check")
    c_check.add_argument("--input", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            errors = validate_contracts([read_json(p) for p in args.task], [read_json(p) for p in args.spec])
            result = {"valid": not errors, "errors": errors, "assurance": "structure and semantic consistency; evidence references are not executed or authenticated"}
            code = int(bool(errors))
        elif args.command == "context":
            result = assemble_context(read_json(args.input), args.budget_tokens, args.reserve_tokens); code = 0
        elif args.command == "memory-check":
            result = check_memory(read_json(args.input), args.root); code = int(any(e["status"] != "valid" for e in result))
        elif args.command == "graph-query":
            result = graph_query(read_json(args.input), args.term, args.hops, args.max_nodes, args.max_chars); code = 0
        elif args.command == "research-queries":
            result = generate_research_queries(args.topic, args.domain, args.target); code = 0
        elif args.command == "research-dossier":
            result = validate_research_dossier(read_json(args.input), strict=args.strict); code = int(not result["valid"])
        elif args.command == "debate-eval":
            result = validate_debate_ledger(read_json(args.input), strict=args.strict); code = int(not result["valid"])
        elif args.command == "capability-check":
            result = validate_capability_catalog(read_json(args.input)); code = int(not result["valid"])
        else:
            result = {"error": f"unknown command {args.command}"}; code = 1
        print(json.dumps(result, ensure_ascii=True, indent=2)); return code
    except (ValueError, KeyError, TypeError, OSError, RecursionError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr); return 1


if __name__ == "__main__":
    sys.exit(main())
