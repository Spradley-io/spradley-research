import os
import json
from collections import defaultdict
from itertools import combinations

import pipeline_core

OUTPUT_DIR = pipeline_core.OUTPUT_DIR

SEED_RELATIONSHIP_TYPES = ["CAUSES", "ENABLES", "IS_BARRIER_TO", "TENSION_WITH", "OPPOSES"]

CONFIG = {
    "RELATIONSHIP_VOCAB_CAP":    8,
    "TYPED_EDGES_TOTAL_CAP":     12,
    "MIN_EDGE_SPEAKER_SHARE":    0.20,
    "CO_OCCURRENCE_MIN_SHARE":   0.20,
    "UPLOAD_CO_EDGES":           False,
    "RELATIONSHIP_MODE":         "per_interview",
    "DATASET_ID":                "the-office-2",
    "LLM_TEMPERATURE":           0.1,
    "MIN_ENTITY_NODES":          4,
    "MAX_ENTITY_NODES":          7,
    "MIN_ENTITY_SPEAKER_COUNT":  2,
    "MAX_EDGES_PER_PAIR":        2,
}


# ── JSON helpers ─────────────────────────────────────────────────────────────

def _load_json(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(data, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_pipeline_outputs() -> dict:
    return {
        "clusters":        _load_json("clusters.json"),
        "lineage":         _load_json("lineage.json"),
        "interview_store": _load_json("interview_store.json"),
        "db":              _load_json("db.json"),
        "experiments":     _load_json("experiments.json"),
        "run_meta":        _load_json("run_meta.json"),
    }


def get_n(run_meta: dict, interview_store: dict) -> int:
    return run_meta.get("n_interviews") or len(interview_store)


# ── G1: Build nodes ──────────────────────────────────────────────────────────

def build_interview_cluster_map(lineage: dict) -> dict:
    """Returns {interview_id: set(cluster_names)} from lineage l2_codes."""
    mapping = defaultdict(set)
    for cluster_name, cluster_data in lineage.items():
        for l2 in cluster_data.get("l2_codes", []):
            iid = l2.get("interview_id")
            if iid:
                mapping[iid].add(cluster_name)
    return dict(mapping)


def build_cluster_nodes(clusters: dict, lineage: dict, n: int) -> list:
    nodes = []
    for name, data in clusters.items():
        support_count = len(lineage.get(name, {}).get("l1_qa_ids", []))
        speaker_count = data.get("voice_count", 0)
        nodes.append({
            "id":            name,
            "name":          name,
            "category":      data.get("category", ""),
            "tagline":       data.get("tagline", ""),
            "summary":       data.get("summary", ""),
            "quotes":        data.get("quotes", []),
            "tag":           data.get("tag", ""),
            "support_count": support_count,
            "speaker_count": speaker_count,
            "spread":        round(speaker_count / n, 3) if n else 0,
            "hub_score":     0.0,
        })
    return nodes


def build_experiment_nodes(experiments: list, clusters: dict) -> tuple:
    """Returns (experiment_nodes, addresses_edges).

    Requires each experiment to have a 'source_cluster' key.
    Raises ValueError if the field is absent -- re-run C9 and C11 in
    Pipeline_Execution.ipynb to regenerate experiments.json with that field.
    """
    if experiments and "source_cluster" not in experiments[0]:
        raise ValueError(
            "experiments.json is missing 'source_cluster'. "
            "Re-run cells C9 and C11 in Pipeline_Execution.ipynb first."
        )

    exp_nodes = []
    addresses_edges = []
    for exp in experiments:
        slug = exp["title"].lower().replace(" ", "-")[:40]
        exp_nodes.append({
            "id":                slug,
            "action":            exp.get("title", ""),
            "finding":           exp.get("insight", ""),
            "try_this":          exp.get("try_this", ""),
            "success_criterion": exp.get("working_when", ""),
            "tag":               exp.get("tag", ""),
        })
        src = exp.get("source_cluster", "")
        if src and src in clusters:
            addresses_edges.append({
                "experiment_id": slug,
                "cluster_id":    src,
            })

    return exp_nodes, addresses_edges


# ── G2b: Entity extraction (LLM) ─────────────────────────────────────────────

_PROMPT_ENTITY = """\
You are analysing anonymised employee interview data.
Below are all Q&A pairs grouped by interview.

Identify between {min_n} and {max_n} of the most analytically significant entities across \
these interviews -- prioritise entities that help explain the research findings, not just \
entities mentioned frequently.
Entities can be: roles/titles (use role not real name), recurring processes, tools/systems, \
or organisational units.

Research themes (exact names):
{cluster_names}

Experiments proposed (exact titles):
{experiment_titles}

Q&A evidence:
{qa_text}

For each entity, describe all meaningful relationships it has to:
  (a) research themes (cluster names above)
  (b) experiments (experiment titles above)
  (c) other entities in your own list

Use whatever relationship type best describes the actual dynamic -- e.g. CAUSES, IS_BARRIER_TO,
ENABLES, CREATES_TENSION_IN, UNDERMINES, MOTIVATES_NEED_FOR, IS_SEEN_AS, etc.
You may propose up to {max_edges_per_pair} distinct relationship types per pair where the data \
is genuinely ambiguous or contradictory (e.g. the same person both ENABLES a coping strategy \
AND CAUSES the stress that requires it).

Return JSON (between {min_n} and {max_n} entries, ordered by analytical relevance):
[{{"name": "<entity name>",
   "type": "person|process|tool|org_unit",
   "description": "<one sentence>",
   "speaker_count": <int>,
   "relationships": [
     {{"target": "<exact cluster name | exact experiment title | entity name>",
       "target_type": "cluster|experiment|entity",
       "type": "<RELATIONSHIP_TYPE>",
       "rationale": "<one sentence>",
       "speaker_count": <int>}}
   ]}}]

Rules:
- Only include entities mentioned by at least {min_speakers} distinct speakers
- Use role titles only -- no real names
- target values for clusters and experiments must be exact strings from the lists above
- entity-to-entity targets must be names of other entities in your own returned list\
"""


def extract_entity_nodes(db: dict, clusters: dict, lineage: dict, n: int,
                          exp_nodes: list = None) -> tuple:
    # Assemble Q&A text grouped by interview prefix
    interviews = defaultdict(list)
    for qa_id, entry in db.items():
        prefix = qa_id[:8]
        interviews[prefix].append(
            f"[{qa_id}] Q: {entry['question']}\nA: {entry['anonymised_answer']}"
        )
    qa_text = ""
    for iid, turns in sorted(interviews.items()):
        qa_text += f"\n--- Interview {iid} ---\n" + "\n\n".join(turns) + "\n"

    cluster_names_str = "\n".join(f"- {name}" for name in sorted(clusters.keys()))
    exp_list          = exp_nodes or []
    exp_titles_str    = "\n".join(f"- {e['action']}" for e in exp_list) or "(none)"
    exp_title_to_id   = {e["action"]: e["id"] for e in exp_list}

    prompt = _PROMPT_ENTITY.format(
        min_n=CONFIG["MIN_ENTITY_NODES"],
        max_n=CONFIG["MAX_ENTITY_NODES"],
        cluster_names=cluster_names_str,
        experiment_titles=exp_titles_str,
        qa_text=qa_text.strip(),
        max_edges_per_pair=CONFIG["MAX_EDGES_PER_PAIR"],
        min_speakers=CONFIG["MIN_ENTITY_SPEAKER_COUNT"],
    )
    raw = pipeline_core.call_llm(prompt, temperature=CONFIG["LLM_TEMPERATURE"])
    try:
        items = pipeline_core.parse_json_safe(raw)
    except Exception:
        items = []

    valid_clusters = set(clusters.keys())
    entity_names   = {item.get("name", "").strip() for item in items if item.get("name")}

    entity_nodes_out = []
    entity_edges_out = []

    for item in items:
        name = item.get("name", "").strip()
        if not name:
            continue
        slug          = name.lower().replace(" ", "-")[:40]
        speaker_count = item.get("speaker_count", 0)
        connected_clusters = [
            r["target"] for r in item.get("relationships", [])
            if r.get("target_type") == "cluster" and r.get("target") in valid_clusters
        ]
        entity_nodes_out.append({
            "id":                slug,
            "name":              name,
            "type":              item.get("type", "org_unit"),
            "description":       item.get("description", ""),
            "speaker_count":     speaker_count,
            "spread":            round(speaker_count / n, 3) if n else 0,
            "connected_clusters": connected_clusters,
        })

        pair_counts = defaultdict(int)
        for rel in item.get("relationships", []):
            target      = rel.get("target", "")
            target_type = rel.get("target_type", "")
            rel_type    = rel.get("type", "RELATES_TO").upper().replace(" ", "_")
            rel_sc      = rel.get("speaker_count", 0)

            if target_type == "cluster":
                if target not in valid_clusters:
                    continue
                target_id = target
            elif target_type == "experiment":
                if target not in exp_title_to_id:
                    continue
                target_id = exp_title_to_id[target]
            elif target_type == "entity":
                if target not in entity_names or target == name:
                    continue
                target_id = target.lower().replace(" ", "-")[:40]
            else:
                continue

            pair_key = (slug, target_id, target_type)
            if pair_counts[pair_key] >= CONFIG["MAX_EDGES_PER_PAIR"]:
                continue
            pair_counts[pair_key] += 1

            entity_edges_out.append({
                "source_id":     slug,
                "target_id":     target_id,
                "target_type":   target_type,
                "type":          rel_type,
                "rationale":     rel.get("rationale", ""),
                "speaker_count": rel_sc,
            })

    return entity_nodes_out, entity_edges_out


# ── G2: Co-occurrence edges ───────────────────────────────────────────────────

def build_co_occurrence_edges(interview_cluster_map: dict, n: int) -> list:
    co = defaultdict(set)
    for iid, cluster_set in interview_cluster_map.items():
        for a, b in combinations(sorted(cluster_set), 2):
            co[(a, b)].add(iid)

    min_share = CONFIG["CO_OCCURRENCE_MIN_SHARE"]
    edges = []
    for (a, b), persons in co.items():
        weight = len(persons) / n if n else 0
        if weight > min_share:
            edges.append({
                "a":            a,
                "b":            b,
                "weight":       round(weight, 3),
                "person_count": len(persons),
            })
    edges.sort(key=lambda e: e["weight"], reverse=True)
    return edges


# ── G3: Per-interview typed proposals (LLM) ──────────────────────────────────

_PROMPT_G3 = """\
You are analysing one employee interview to find directed relationships among the themes they raised.

Themes present in this employee's interview (name + tagline):
{cluster_list}

Their anonymised Q&A evidence:
{qa_evidence}

Propose directed relationships between any two of these themes where this employee's answers \
suggest a causal, enabling, barrier, tension, or opposition link.

Examples of relationship types: {seed_types}.
Derive your own type label if it better describes the actual dynamic -- use the most precise \
term the data supports.

Return a JSON list (return [] if no clear relationship exists):
[{{"source_cluster": "<exact cluster name>", "target_cluster": "<exact cluster name>",
   "type_raw": "<RELATIONSHIP_TYPE>", "rationale": "<one sentence>",
   "evidence_turn_ids": ["<turn_id>"]}}]

Rules: source and target must differ. Use cluster names exactly as given above.
Only propose a relationship if the answers explain HOW or WHY one theme influences the other -- \
do NOT propose just because two themes appeared in the same interview.\
"""

_PROMPT_G3_GLOBAL = """\
You are analysing a set of thematic clusters from employee interviews to propose directed \
relationships between them, informed by co-occurrence patterns.

Clusters (name, tagline, category):
{cluster_summaries}

Co-occurrence hints (pairs that appear together for at least {min_share:.0%} of speakers):
{co_hints}

Propose directed typed relationships where the cluster data suggests causal, enabling, \
barrier, tension, or opposition links.

Examples of relationship types: {seed_types}.
Derive your own type label if it better describes the actual dynamic -- use the most precise \
term the data supports. Only propose where there is a clear directional mechanism, not just \
co-occurrence.

Return a JSON list ([] if nothing is clear):
[{{"source_cluster": "<exact name>", "target_cluster": "<exact name>",
   "type_raw": "<TYPE>", "rationale": "<one sentence>",
   "evidence_turn_ids": []}}]

Use cluster names exactly as given.\
"""


def _qa_evidence_for_interview(interview_id: str, cluster_names: set,
                                lineage: dict, db: dict) -> str:
    prefix = interview_id.replace("-", "")[:8]
    relevant_ids = set()
    for cname in cluster_names:
        for qa_id in lineage.get(cname, {}).get("l1_qa_ids", []):
            if qa_id.startswith(prefix):
                relevant_ids.add(qa_id)
    lines = []
    for qa_id in sorted(relevant_ids):
        entry = db.get(qa_id)
        if entry:
            lines.append(f"[{qa_id}] Q: {entry['question']}\nA: {entry['anonymised_answer']}")
    return "\n\n".join(lines) if lines else "(no evidence found)"


def run_g3_proposals(interview_cluster_map: dict, cluster_nodes_by_name: dict,
                     lineage: dict, db: dict, co_edges: list = None) -> list:
    if CONFIG.get("RELATIONSHIP_MODE") == "global":
        return _run_g3_global(list(cluster_nodes_by_name.values()), co_edges or [])

    proposals = []
    for iid, cluster_set in interview_cluster_map.items():
        if len(cluster_set) < 2:
            continue
        cluster_list = "\n".join(
            f"- {n}: {cluster_nodes_by_name[n]['tagline']}"
            for n in sorted(cluster_set) if n in cluster_nodes_by_name
        )
        qa_evidence = _qa_evidence_for_interview(iid, cluster_set, lineage, db)
        prompt = _PROMPT_G3.format(
            cluster_list=cluster_list,
            qa_evidence=qa_evidence,
            seed_types=", ".join(SEED_RELATIONSHIP_TYPES),
        )
        raw = pipeline_core.call_llm(prompt, temperature=CONFIG["LLM_TEMPERATURE"])
        try:
            items = pipeline_core.parse_json_safe(raw)
        except Exception:
            items = []
        valid = set(cluster_nodes_by_name)
        for item in items:
            src = item.get("source_cluster", "")
            tgt = item.get("target_cluster", "")
            if src in valid and tgt in valid and src != tgt:
                proposals.append({
                    "interview_id":      iid,
                    "person_id":         iid,
                    "source_cluster":    src,
                    "target_cluster":    tgt,
                    "type_raw":          item.get("type_raw", "RELATES_TO"),
                    "rationale":         item.get("rationale", ""),
                    "evidence_turn_ids": item.get("evidence_turn_ids", []),
                })
    return proposals


def _run_g3_global(cluster_nodes: list, co_edges: list) -> list:
    cluster_summaries = "\n".join(
        f"- {n['name']} ({n['category']}): {n['tagline']}" for n in cluster_nodes
    )
    co_hints = "\n".join(
        f"- {e['a']} <-> {e['b']} (weight {e['weight']})" for e in co_edges
    ) or "(none)"
    prompt = _PROMPT_G3_GLOBAL.format(
        cluster_summaries=cluster_summaries,
        co_hints=co_hints,
        min_share=CONFIG["CO_OCCURRENCE_MIN_SHARE"],
        seed_types=", ".join(SEED_RELATIONSHIP_TYPES),
    )
    raw = pipeline_core.call_llm(prompt, temperature=CONFIG["LLM_TEMPERATURE"])
    try:
        items = pipeline_core.parse_json_safe(raw)
    except Exception:
        items = []
    valid = {n["name"] for n in cluster_nodes}
    proposals = []
    for item in items:
        src = item.get("source_cluster", "")
        tgt = item.get("target_cluster", "")
        if src in valid and tgt in valid and src != tgt:
            proposals.append({
                "interview_id":      "global",
                "person_id":         "global",
                "source_cluster":    src,
                "target_cluster":    tgt,
                "type_raw":          item.get("type_raw", "RELATES_TO"),
                "rationale":         item.get("rationale", ""),
                "evidence_turn_ids": [],
            })
    return proposals


# ── G4: Consolidate proposals ────────────────────────────────────────────────

_PROMPT_G4_RENAME = """\
Canonicalise these relationship type labels. Merge near-synonyms into single labels.
Return at most {vocab_cap} distinct types. Prefer types from this seed list: {seed_types}.

Types to normalise (JSON list): {raw_types}

Return a JSON object only: {{"<type_raw>": "<type_canonical>", ...}}\
"""


def canonicalize_types(proposals: list) -> dict:
    raw_types = sorted({p["type_raw"] for p in proposals})
    if not raw_types:
        return {}
    prompt = _PROMPT_G4_RENAME.format(
        vocab_cap=CONFIG["RELATIONSHIP_VOCAB_CAP"],
        seed_types=", ".join(SEED_RELATIONSHIP_TYPES),
        raw_types=json.dumps(raw_types),
    )
    raw = pipeline_core.call_llm(prompt, temperature=0.0)
    return pipeline_core.parse_json_safe(raw)


def count_groups(proposals: list, type_mapping: dict, n: int) -> list:
    groups = defaultdict(lambda: {"persons": set(), "rationales": [], "evidence": []})
    for p in proposals:
        canonical = type_mapping.get(p["type_raw"], p["type_raw"])
        key = (p["source_cluster"], p["target_cluster"], canonical)
        groups[key]["persons"].add(p["person_id"])
        groups[key]["rationales"].append(p["rationale"])
        groups[key]["evidence"].extend(p.get("evidence_turn_ids", []))

    result = []
    for (src, tgt, typ), g in groups.items():
        weight = len(g["persons"]) / n if n else 0
        result.append({
            "id":             f"{src}::{typ}::{tgt}",
            "source_cluster": src,
            "target_cluster": tgt,
            "type":           typ,
            "persons":        list(g["persons"]),
            "person_count":   len(g["persons"]),
            "weight":         round(weight, 3),
            "rationales":     g["rationales"],
            "evidence":       list(set(g["evidence"])),
        })
    return result


# ── G5: Finalize edges ───────────────────────────────────────────────────────

_PROMPT_G5_CONTEXT = """\
Write a one-sentence insight for each relationship edge below. Ground each sentence in the
rationale(s) provided. Match the tone of this example:
"Teams want more say in planning yet feel overwhelmed by meetings -- the lever is quality of
involvement, not quantity."

Edges (JSON):
{edges_json}

Return a JSON list:
[{{"id": "<edge_id>", "context": "<one sentence>", "is_paradox": true|false}}]

is_paradox is true if the type is TENSION_WITH or OPPOSES, or if the relationship is a \
genuine trade-off between two things that both matter.\
"""


def finalize_typed_edges(groups: list) -> list:
    min_share    = CONFIG["MIN_EDGE_SPEAKER_SHARE"]
    total_cap    = CONFIG["TYPED_EDGES_TOTAL_CAP"]
    max_per_pair = CONFIG.get("MAX_EDGES_PER_PAIR", 2)

    # Drop below minimum support (strict greater than)
    surviving = [g for g in groups if g["weight"] > min_share]

    # Per ordered pair: keep up to MAX_EDGES_PER_PAIR distinct types by weight
    seen = defaultdict(list)
    for g in sorted(surviving, key=lambda x: x["weight"], reverse=True):
        key = (g["source_cluster"], g["target_cluster"])
        if len(seen[key]) < max_per_pair:
            seen[key].append(g)
    surviving = [g for gs in seen.values() for g in gs]

    # Cap total
    surviving.sort(key=lambda x: x["weight"], reverse=True)
    surviving = surviving[:total_cap]

    if not surviving:
        return []

    # LLM: context + is_paradox for each surviving edge
    edges_for_prompt = [
        {
            "id":         g["id"],
            "source":     g["source_cluster"],
            "target":     g["target_cluster"],
            "type":       g["type"],
            "rationales": g["rationales"],
        }
        for g in surviving
    ]
    prompt = _PROMPT_G5_CONTEXT.format(
        edges_json=json.dumps(edges_for_prompt, indent=2)
    )
    raw = pipeline_core.call_llm(prompt, temperature=CONFIG["LLM_TEMPERATURE"])
    try:
        context_list = pipeline_core.parse_json_safe(raw)
    except Exception:
        context_list = []

    context_by_id = {c["id"]: c for c in context_list}
    for g in surviving:
        ctx = context_by_id.get(g["id"], {})
        g["context"]    = ctx.get("context", "")
        g["is_paradox"] = ctx.get("is_paradox",
                                  g["type"] in {"TENSION_WITH", "OPPOSES"})
    return surviving


def flag_hub_scores(cluster_nodes: list, typed_edges: list) -> list:
    """Sets hub_score on each cluster node: fraction of typed edges it appears in."""
    total = len(typed_edges)
    counts = defaultdict(int)
    for e in typed_edges:
        counts[e["source_cluster"]] += 1
        counts[e["target_cluster"]] += 1
    for node in cluster_nodes:
        node["hub_score"] = round(counts[node["name"]] / total, 3) if total else 0.0
    return cluster_nodes


# ── Assemble + save ──────────────────────────────────────────────────────────

def assemble_graph(cluster_nodes: list, exp_nodes: list, co_edges: list,
                   typed_edges: list, addresses_edges: list,
                   entity_nodes: list = None, entity_edges: list = None) -> dict:
    canonical_types = sorted({e["type"] for e in typed_edges})
    return {
        "dataset":         CONFIG["DATASET_ID"],
        "config_snapshot": dict(CONFIG),
        "nodes":           cluster_nodes,
        "experiments":     exp_nodes,
        "co_edges":        co_edges,
        "typed_edges":     typed_edges,
        "addresses_edges": addresses_edges,
        "entity_nodes":    entity_nodes or [],
        "entity_edges":    entity_edges or [],
        "canonical_types": canonical_types,
    }


def save_graph(graph: dict):
    _save_json(graph, "graph.json")
    print(
        f"graph.json saved  "
        f"({len(graph['nodes'])} clusters, "
        f"{len(graph['co_edges'])} co-edges, "
        f"{len(graph['typed_edges'])} typed edges, "
        f"{len(graph.get('entity_nodes', []))} entities)"
    )


# ── G6: Neo4j load ───────────────────────────────────────────────────────────

def _check_apoc(session) -> bool:
    try:
        result = session.run("RETURN apoc.version() AS v")
        result.single()
        return True
    except Exception:
        return False


def load_to_neo4j(graph: dict):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("neo4j driver not installed. Run: pip install neo4j")

    uri      = os.environ["NEO4J_URI"]
    user     = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    dataset  = graph["dataset"]

    # Corporate TLS inspection replaces the Aura certificate, so switch to the
    # +ssc scheme (self-signed cert) which keeps TLS but skips verification --
    # equivalent to httpx verify=False used for the Anthropic client.
    uri = uri.replace("neo4j+s://", "neo4j+ssc://").replace("bolt+s://", "bolt+ssc://")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    counts = {
        "clusters":     0,
        "experiments":  0,
        "co_edges":     0,
        "addresses":    0,
        "typed_edges":  0,
        "entities":     0,
        "entity_edges": 0,
    }

    _LABEL_MAP = {"cluster": "Cluster", "experiment": "Experiment", "entity": "Entity"}

    with driver.session() as session:
        # Scoped clean -- only this dataset
        session.run("MATCH (n {dataset: $d}) DETACH DELETE n", d=dataset)

        apoc_available = _check_apoc(session)

        # Cluster nodes
        for node in graph["nodes"]:
            session.run(
                """MERGE (c:Cluster {id: $id, dataset: $d})
                   SET c += {name: $name, category: $category, tagline: $tagline,
                              summary: $summary, tag: $tag,
                              support_count: $sc, speaker_count: $spk,
                              spread: $spread, hub_score: $hub}""",
                id=node["id"], d=dataset, name=node["name"],
                category=node["category"], tagline=node["tagline"],
                summary=node["summary"], tag=node["tag"],
                sc=node["support_count"], spk=node["speaker_count"],
                spread=node["spread"], hub=node.get("hub_score", 0.0),
            )
            counts["clusters"] += 1

        # Experiment nodes
        for exp in graph.get("experiments", []):
            session.run(
                """MERGE (e:Experiment {id: $id, dataset: $d})
                   SET e += {action: $action, finding: $finding,
                              try_this: $try_this, success_criterion: $sc, tag: $tag}""",
                id=exp["id"], d=dataset, action=exp["action"],
                finding=exp["finding"], try_this=exp["try_this"],
                sc=exp["success_criterion"], tag=exp["tag"],
            )
            counts["experiments"] += 1

        # CO_OCCURS_WITH -- skipped by default (UPLOAD_CO_EDGES: False)
        if graph.get("config_snapshot", {}).get("UPLOAD_CO_EDGES", True):
            for edge in graph["co_edges"]:
                session.run(
                    """MATCH (a:Cluster {id: $a, dataset: $d}),
                             (b:Cluster {id: $b, dataset: $d})
                       MERGE (a)-[r:CO_OCCURS_WITH]-(b)
                       SET r.weight = $w, r.person_count = $pc""",
                    a=edge["a"], b=edge["b"], d=dataset,
                    w=edge["weight"], pc=edge["person_count"],
                )
                counts["co_edges"] += 1

        # ADDRESSES (Experiment -> Cluster)
        for ae in graph.get("addresses_edges", []):
            session.run(
                """MATCH (e:Experiment {id: $eid, dataset: $d}),
                         (c:Cluster {id: $cid, dataset: $d})
                   MERGE (e)-[:ADDRESSES]->(c)""",
                eid=ae["experiment_id"], cid=ae["cluster_id"], d=dataset,
            )
            counts["addresses"] += 1

        # Typed directed cluster-cluster edges -- APOC preferred, RELATES_TO fallback
        for edge in graph["typed_edges"]:
            if apoc_available:
                session.run(
                    """MATCH (a:Cluster {id: $src, dataset: $d}),
                             (b:Cluster {id: $tgt, dataset: $d})
                       CALL apoc.merge.relationship(
                           a, $type, {},
                           {context: $ctx, weight: $w, is_paradox: $ip, evidence: $ev},
                           b) YIELD rel RETURN rel""",
                    src=edge["source_cluster"], tgt=edge["target_cluster"], d=dataset,
                    type=edge["type"], ctx=edge.get("context", ""),
                    w=edge["weight"], ip=edge.get("is_paradox", False),
                    ev=edge.get("evidence", []),
                )
            else:
                session.run(
                    """MATCH (a:Cluster {id: $src, dataset: $d}),
                             (b:Cluster {id: $tgt, dataset: $d})
                       MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
                       SET r.context = $ctx, r.weight = $w,
                           r.is_paradox = $ip, r.evidence = $ev""",
                    src=edge["source_cluster"], tgt=edge["target_cluster"], d=dataset,
                    type=edge["type"], ctx=edge.get("context", ""),
                    w=edge["weight"], ip=edge.get("is_paradox", False),
                    ev=edge.get("evidence", []),
                )
            counts["typed_edges"] += 1

        # Entity nodes
        for ent in graph.get("entity_nodes", []):
            session.run(
                """MERGE (e:Entity {id: $id, dataset: $d})
                   SET e += {name: $name, type: $type, description: $desc,
                              speaker_count: $sc, spread: $spread}""",
                id=ent["id"], d=dataset, name=ent["name"], type=ent["type"],
                desc=ent["description"], sc=ent["speaker_count"], spread=ent["spread"],
            )
            counts["entities"] += 1

        # Typed entity edges -- target can be Cluster, Experiment, or Entity
        for ee in graph.get("entity_edges", []):
            tgt_label = _LABEL_MAP.get(ee["target_type"], "Cluster")
            if apoc_available:
                session.run(
                    f"""MATCH (src:Entity {{id: $sid, dataset: $d}}),
                             (tgt:{tgt_label} {{id: $tid, dataset: $d}})
                       CALL apoc.merge.relationship(src, $type, {{}},
                           {{rationale: $rationale, speaker_count: $sc}}, tgt)
                       YIELD rel RETURN rel""",
                    sid=ee["source_id"], tid=ee["target_id"], d=dataset,
                    type=ee["type"], rationale=ee.get("rationale", ""),
                    sc=ee["speaker_count"],
                )
            else:
                session.run(
                    f"""MATCH (src:Entity {{id: $sid, dataset: $d}}),
                             (tgt:{tgt_label} {{id: $tid, dataset: $d}})
                       MERGE (src)-[r:RELATES_TO {{type: $type}}]->(tgt)
                       SET r.rationale = $rationale, r.speaker_count = $sc""",
                    sid=ee["source_id"], tid=ee["target_id"], d=dataset,
                    type=ee["type"], rationale=ee.get("rationale", ""),
                    sc=ee["speaker_count"],
                )
            counts["entity_edges"] += 1

        # Verification
        n_exp_db  = session.run(
            "MATCH (e:Experiment {dataset: $d}) RETURN count(e) AS n", d=dataset
        ).single()["n"]
        n_addr_db = session.run(
            "MATCH ()-[r:ADDRESSES]->({dataset: $d}) RETURN count(r) AS n", d=dataset
        ).single()["n"]

    driver.close()
    mode = "APOC dynamic types" if apoc_available else "RELATES_TO fallback (APOC not available)"
    print(f"Neo4j load complete  [{mode}]")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"\nVerification:")
    print(f"  experiments in Neo4j: {n_exp_db}  "
          f"(graph has {len(graph.get('experiments', []))})")
    print(f"  ADDRESSES edges:      {n_addr_db}  "
          f"(graph has {len(graph.get('addresses_edges', []))})")
