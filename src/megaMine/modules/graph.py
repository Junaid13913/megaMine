"""
graph.py — megaMine v2.0
Knowledge Graph Construction Module

PURPOSE:
    Converts megaMine output into a proper biomedical
    knowledge graph where Evidence and Publications are
    first-class nodes — not just edge weights.

    This enables provenance-aware graph queries:
    "Show drugs targeting EGFR in NSCLC where resistance
     was observed in RCT studies published after 2019"

    Simple weighted graphs cannot answer this.
    megaMine KG can — because evidence is traceable.

NODE TYPES:
    Gene        — HGNC-validated gene symbol
    Drug        — normalized drug name
    Cancer      — normalized cancer type
    Alteration  — specific mutation/alteration
    Evidence    — single extracted evidence item
    Publication — PubMed paper (PMID)

EDGE TYPES:
    Gene      -[EXPRESSED_IN]->    Cancer
    Drug      -[TREATS]->          Cancer
    Drug      -[TARGETS]->         Gene
    Alteration-[CONFERS_RESISTANCE_TO]-> Drug
    Alteration-[SENSITIZES_TO]->   Drug
    Evidence  -[SUPPORTS]->        Gene
    Evidence  -[SUPPORTS]->        Drug
    Evidence  -[SUPPORTS]->        Cancer
    Publication-[CONTAINS]->       Evidence

AUTHOR: Muhammad Junaid
VERSION: 2.0.0
"""

import pandas as pd
import json
from typing import Optional, Dict, List
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️  networkx not installed — install with: pip install networkx")


# ─── Node type constants ───────────────────────────────────────
NODE_GENE        = "Gene"
NODE_DRUG        = "Drug"
NODE_CANCER      = "Cancer"
NODE_ALTERATION  = "Alteration"
NODE_EVIDENCE    = "Evidence"
NODE_PUBLICATION = "Publication"

# ─── Edge type constants ───────────────────────────────────────
EDGE_EXPRESSED_IN          = "ASSOCIATED_WITH"   # conservative: literature does not prove expression
EDGE_TREATS                = "EVALUATED_IN"      # conservative: literature does not prove approved treatment
EDGE_TARGETS               = "TARGETS"
EDGE_CONFERS_RESISTANCE_TO = "CONFERS_RESISTANCE_TO"
EDGE_SENSITIZES_TO         = "SENSITIZES_TO"
EDGE_SUPPORTS              = "SUPPORTS"
EDGE_CONTAINS              = "CONTAINS"


def _clean(s: str) -> str:
    """Clean and normalize a string for use as node ID."""
    if not s:
        return "unknown"
    return str(s).strip().lower().replace(" ", "_").replace(";", "").replace("/", "_")


def _gene_id(gene: str) -> str:
    return f"gene:{gene.upper().strip()}"


def _drug_id(drug: str) -> str:
    return f"drug:{_clean(drug)}"


def _cancer_id(cancer: str) -> str:
    # Use first part before semicolon for cleaner IDs
    c = cancer.split(";")[0].strip() if cancer else "unknown"
    return f"cancer:{_clean(c)}"


def _alteration_id(gene: str, alteration: str) -> str:
    return f"alt:{gene.upper()}_{_clean(alteration)}"


def _evidence_id(gene: str, drug: str, cancer: str, pmid: str, idx: int) -> str:
    return f"ev:{gene.upper()}_{_clean(drug)}_{_clean(cancer.split(';')[0])}_{pmid}_{idx}"


def _pub_id(pmid: str) -> str:
    return f"pub:pmid_{pmid}"


def build_graph(
    df: pd.DataFrame,
    temporal_df: Optional[pd.DataFrame] = None,
    contradiction_df: Optional[pd.DataFrame] = None,
) -> "nx.MultiDiGraph":
    """
    Build a provenance-aware knowledge graph from megaMine output.

    Every Evidence item is a node — not just a weight.
    Every Publication is a node — fully traceable.

    Parameters
    ----------
    df : pd.DataFrame
        Raw megaMine output from extractor.py
    temporal_df : pd.DataFrame, optional
        Trend classifications from temporal.py
    contradiction_df : pd.DataFrame, optional
        Contradiction flags from contradiction.py

    Returns
    -------
    nx.MultiDiGraph
        Directed multigraph with all node and edge types
    """
    if not HAS_NETWORKX:
        raise ImportError("Install networkx: pip install networkx")

    G = nx.MultiDiGraph()
    G.graph["name"]    = "megaMine Knowledge Graph"
    G.graph["version"] = "2.0.0"
    G.graph["schema"]  = "provenance_aware_v1"

    print("🕸️  Building knowledge graph...")

    # Build temporal lookup for quick access
    temporal_lookup = {}
    if temporal_df is not None and len(temporal_df) > 0:
        for _, row in temporal_df.iterrows():
            key = (
                row["biomarker"].upper(),
                row["drug_primary"].lower(),
                row["cancer_type"].split(";")[0].strip().lower()
            )
            temporal_lookup[key] = row.get("temporal_trend", "")

    # Build contradiction lookup
    contradiction_lookup = {}
    if contradiction_df is not None and len(contradiction_df) > 0:
        for _, row in contradiction_df.iterrows():
            key = (
                row["biomarker"].upper(),
                row["drug_primary"].lower(),
                row["cancer_type"].split(";")[0].strip().lower()
            )
            contradiction_lookup[key] = {
                "flag":  row.get("contradiction_flag", ""),
                "score": row.get("conflict_score", 0.0),
                "temporal_conflict": row.get("temporal_conflict", "no"),
            }

    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)

    for idx, row in df.iterrows():
        gene      = str(row.get("biomarker", "") or "").strip()
        drug      = str(row.get("drug_primary", "") or "").strip()
        # Use canonical_cancer_type for clean graph nodes
        cancer    = str(row.get("canonical_cancer_type", "") or
                        row.get("cancer_type", "") or "").strip()
        alteration= str(row.get("alteration", "") or "").strip()
        # Fix: reject nan/none/na alteration values
        if alteration.lower() in {"nan","none","na","n/a","null",""}:
            alteration = ""
        pmid      = str(row.get("pmid", "") or "").strip()
        year      = int(row.get("year", 0) or 0)

        if not gene or not drug or not cancer:
            continue

        # ── Node IDs ──────────────────────────────────────────
        gid  = _gene_id(gene)
        did  = _drug_id(drug)
        cid  = _cancer_id(cancer)
        pid  = _pub_id(pmid)
        evid = _evidence_id(gene, drug, cancer, pmid, idx)

        # ── Triplet-level context ──────────────────────────────
        tkey = (
            gene.upper(),
            drug.lower(),
            cancer.split(";")[0].strip().lower()
        )
        temporal_trend    = temporal_lookup.get(tkey, "")
        contradiction_info= contradiction_lookup.get(tkey, {})

        # ── Add Gene node ──────────────────────────────────────
        if not G.has_node(gid):
            G.add_node(gid,
                node_type  = NODE_GENE,
                label      = gene.upper(),
                gene_type  = str(row.get("gene_type", "") or ""),
            )
            node_counts[NODE_GENE] += 1

        # ── Add Drug node ──────────────────────────────────────
        if not G.has_node(did):
            G.add_node(did,
                node_type     = NODE_DRUG,
                label         = drug,
                therapy_type  = str(row.get("therapy_type", "") or ""),
                accessibility = str(row.get("drug_accessibility", "") or ""),
            )
            node_counts[NODE_DRUG] += 1

        # ── Add Cancer node ────────────────────────────────────
        # Use canonical cancer label — raw_cancer stored as attribute
        if not G.has_node(cid):
            G.add_node(cid,
                node_type         = NODE_CANCER,
                label             = cancer.split(";")[0].strip(),
                histology         = str(row.get("histology", "") or ""),
                raw_cancer_type   = str(row.get("cancer_type", "") or ""),
            )
            node_counts[NODE_CANCER] += 1

        # ── Add Alteration node ────────────────────────────────
        if alteration:
            aid = _alteration_id(gene, alteration)
            if not G.has_node(aid):
                G.add_node(aid,
                    node_type      = NODE_ALTERATION,
                    label          = alteration,
                    alteration_type= str(row.get("alteration_type", "") or ""),
                    gene           = gene.upper(),
                )
                node_counts[NODE_ALTERATION] += 1

        # ── Add Publication node ───────────────────────────────
        if pmid and pmid != "0":
            if not G.has_node(pid):
                G.add_node(pid,
                    node_type   = NODE_PUBLICATION,
                    label       = f"PMID:{pmid}",
                    pmid        = pmid,
                    year        = year,
                    journal     = str(row.get("journal", "") or ""),
                    doi         = str(row.get("doi", "") or ""),
                    study_design= str(row.get("study_design", "") or ""),
                )
                node_counts[NODE_PUBLICATION] += 1

        # ── Add Evidence node ──────────────────────────────────
        # THIS IS THE KEY INNOVATION
        # Evidence is a node, not just a weight
        # Every extracted relationship is traceable
        # Evidence nodes carry full publication + biological metadata
        # This enables ranking, filtering, and provenance queries
        G.add_node(evid,
            node_type           = NODE_EVIDENCE,
            label               = evid,
            # Biological identity
            gene                = str(row.get("biomarker","") or ""),
            biomarker           = str(row.get("biomarker","") or ""),
            alteration          = str(row.get("alteration","") or ""),
            drug_primary        = str(row.get("drug_primary","") or ""),
            canonical_cancer_type = str(row.get("canonical_cancer_type","") or ""),
            # Evidence classification
            evidence_type       = str(row.get("final_evidence_type",
                                      row.get("evidence_type","")) or ""),
            final_evidence_type = str(row.get("final_evidence_type",
                                      row.get("evidence_type","")) or ""),
            evidence_type_raw   = str(row.get("evidence_type","") or ""),
            resistance_observed = str(row.get("resistance_observed","") or ""),
            resistance_evidence = str(row.get("resistance_evidence","") or ""),
            resistance_direction= str(row.get("resistance_direction","") or ""),
            therapeutic_active  = str(row.get("therapeutic_active","") or ""),
            # Publication metadata
            pmid                = str(row.get("pmid","") or ""),
            pmcid               = str(row.get("pmcid","") or ""),
            doi                 = str(row.get("doi","") or ""),
            journal             = str(row.get("journal","") or ""),
            year                = str(row.get("year","") or ""),
            study_design        = str(row.get("study_design","") or ""),
            trial_phase         = str(row.get("trial_phase","") or ""),
            line_of_therapy     = str(row.get("line_of_therapy","") or ""),
            # Verifier metadata
            llm_confidence      = row.get("llm_confidence", 0),
            llm_verified        = str(row.get("llm_verified","") or ""),
            # Triplet-level enrichment from other modules
            temporal_trend     = temporal_trend,
            contradiction_flag = contradiction_info.get("flag", ""),
            conflict_score     = contradiction_info.get("score", 0.0),
        )
        node_counts[NODE_EVIDENCE] += 1

        # ── Add Edges ──────────────────────────────────────────

        # Gene expressed in Cancer
        G.add_edge(gid, cid,
            edge_type = EDGE_EXPRESSED_IN,
            weight    = 1)
        edge_counts[EDGE_EXPRESSED_IN] += 1

        # Drug treats Cancer
        G.add_edge(did, cid,
            edge_type = EDGE_TREATS,
            weight    = 1)
        edge_counts[EDGE_TREATS] += 1

        # Drug targets Gene
        G.add_edge(did, gid,
            edge_type = EDGE_TARGETS,
            weight    = 1)
        edge_counts[EDGE_TARGETS] += 1

        # Alteration edges — resistance or sensitivity
        if alteration:
            aid = _alteration_id(gene, alteration)
            # Use resistance_evidence (precise) not resistance_observed (too broad)
            # resistance_observed was yes for 152/153 rows in real run
            # resistance_evidence is yes only for direct resistance statements
            resist_field = (
                "resistance_evidence"
                if "resistance_evidence" in row.index
                else "resistance_observed"
            )
            if str(row.get(resist_field, "")) == "yes":
                G.add_edge(aid, did,
                    edge_type = EDGE_CONFERS_RESISTANCE_TO,
                    weight    = 1)
                edge_counts[EDGE_CONFERS_RESISTANCE_TO] += 1
            elif str(row.get("therapeutic_active", "")) == "yes":
                G.add_edge(aid, did,
                    edge_type = EDGE_SENSITIZES_TO,
                    weight    = 1)
                edge_counts[EDGE_SENSITIZES_TO] += 1

        # Evidence supports Gene, Drug, Cancer
        G.add_edge(evid, gid,
            edge_type = EDGE_SUPPORTS, weight=1)
        G.add_edge(evid, did,
            edge_type = EDGE_SUPPORTS, weight=1)
        G.add_edge(evid, cid,
            edge_type = EDGE_SUPPORTS, weight=1)
        edge_counts[EDGE_SUPPORTS] += 3

        # Publication contains Evidence
        if pmid and pmid != "0":
            G.add_edge(pid, evid,
                edge_type = EDGE_CONTAINS,
                weight    = 1)
            edge_counts[EDGE_CONTAINS] += 1

    # ── Graph summary ──────────────────────────────────────────
    print(f"\n   ✅ Graph built successfully")
    print(f"\n   📊 Node summary:")
    for ntype, count in sorted(node_counts.items()):
        print(f"      {ntype}: {count:,}")
    print(f"      TOTAL: {G.number_of_nodes():,}")

    print(f"\n   📊 Edge summary:")
    for etype, count in sorted(edge_counts.items()):
        print(f"      {etype}: {count:,}")
    print(f"      TOTAL: {G.number_of_edges():,}")

    return G


def export_graph(
    G: "nx.MultiDiGraph",
    output_prefix: str,
    formats: List[str] = ["graphml", "csv"]
) -> Dict[str, str]:
    """
    Export the knowledge graph to files.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph from build_graph()
    output_prefix : str
        Path prefix for output files
    formats : list
        Export formats: graphml, csv, json

    Returns
    -------
    dict : {format: filepath}
    """
    if not HAS_NETWORKX:
        raise ImportError("Install networkx: pip install networkx")

    outputs = {}

    if "graphml" in formats:
        path = f"{output_prefix}_graph.graphml"
        nx.write_graphml(G, path)
        outputs["graphml"] = path
        print(f"   💾 GraphML: {path}")

    if "csv" in formats:
        # Export node list
        node_path = f"{output_prefix}_nodes.csv"
        node_rows = []
        for node_id, attrs in G.nodes(data=True):
            row = {"node_id": node_id}
            row.update(attrs)
            node_rows.append(row)
        pd.DataFrame(node_rows).to_csv(node_path, index=False)
        outputs["nodes_csv"] = node_path
        print(f"   💾 Nodes CSV: {node_path}")

        # Export edge list
        edge_path = f"{output_prefix}_edges.csv"
        edge_rows = []
        for src, dst, attrs in G.edges(data=True):
            row = {"source": src, "target": dst}
            row.update(attrs)
            edge_rows.append(row)
        pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
        outputs["edges_csv"] = edge_path
        print(f"   💾 Edges CSV: {edge_path}")

    if "json" in formats:
        path = f"{output_prefix}_graph.json"
        data = nx.node_link_data(G)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        outputs["json"] = path
        print(f"   💾 JSON: {path}")

    return outputs


def query_graph(
    G: "nx.MultiDiGraph",
    gene: Optional[str] = None,
    drug: Optional[str] = None,
    cancer: Optional[str] = None,
    evidence_type: Optional[str] = None,
    resistance_only: bool = False,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query the knowledge graph with filters.
    Returns matching Evidence nodes with full context.

    This is the key advantage over weighted graphs —
    you can filter by ANY evidence property.

    Examples
    --------
    # All resistance evidence for EGFR
    query_graph(G, gene="EGFR", resistance_only=True)

    # RCT evidence for erlotinib after 2019
    query_graph(G, drug="erlotinib", min_year=2019)

    # All evidence for EGFR + erlotinib in NSCLC
    query_graph(G, gene="EGFR", drug="erlotinib", cancer="NSCLC")
    """
    if not HAS_NETWORKX:
        raise ImportError("Install networkx: pip install networkx")

    results = []

    for node_id, attrs in G.nodes(data=True):
        if attrs.get("node_type") != NODE_EVIDENCE:
            continue

        # Apply filters
        # Use resistance_evidence if available — more precise
        resist_field = "resistance_evidence" if "resistance_evidence" in attrs else "resistance_observed"
        if resistance_only and attrs.get(resist_field) != "yes":
            continue
        if evidence_type and attrs.get("evidence_type") != evidence_type:
            continue
        if min_year and int(attrs.get("year", 0) or 0) < min_year:
            continue

        # Get connected Gene, Drug, Cancer nodes
        connected_genes   = []
        connected_drugs   = []
        connected_cancers = []
        source_pmid       = ""

        for src, dst, eattrs in G.edges(node_id, data=True):
            target_attrs = G.nodes[dst]
            ntype = target_attrs.get("node_type", "")
            if ntype == NODE_GENE and (not gene or
               target_attrs.get("label","").upper() == gene.upper()):
                connected_genes.append(target_attrs.get("label",""))
            elif ntype == NODE_DRUG and (not drug or
               target_attrs.get("label","").lower() == drug.lower()):
                connected_drugs.append(target_attrs.get("label",""))
            elif ntype == NODE_CANCER and (not cancer or
               cancer.lower() in target_attrs.get("label","").lower()):
                connected_cancers.append(target_attrs.get("label",""))

        # Get publication
        for src, dst, eattrs in G.in_edges(node_id, data=True):
            if G.nodes[src].get("node_type") == NODE_PUBLICATION:
                source_pmid = G.nodes[src].get("pmid", "")

        # Filter — must match requested gene/drug/cancer
        if gene and not connected_genes:
            continue
        if drug and not connected_drugs:
            continue
        if cancer and not connected_cancers:
            continue

        results.append({
            "evidence_id":        node_id,
            "genes":              "; ".join(connected_genes),
            "drugs":              "; ".join(connected_drugs),
            "cancers":            "; ".join(connected_cancers),
            "evidence_type":      attrs.get("evidence_type", ""),
            "resistance_observed":attrs.get("resistance_observed", ""),
            "therapeutic_active": attrs.get("therapeutic_active", ""),
            "study_design":       attrs.get("study_design", ""),
            "year":               attrs.get("year", ""),
            "temporal_trend":     attrs.get("temporal_trend", ""),
            "contradiction_flag": attrs.get("contradiction_flag", ""),
            "pmid":               source_pmid,
        })

    return pd.DataFrame(results)


def get_graph_stats(G: "nx.MultiDiGraph") -> dict:
    """
    Get summary statistics for the knowledge graph.
    Useful for the paper methods section.
    """
    stats = {
        "total_nodes":        G.number_of_nodes(),
        "total_edges":        G.number_of_edges(),
        "gene_nodes":         sum(1 for _,a in G.nodes(data=True) if a.get("node_type")==NODE_GENE),
        "drug_nodes":         sum(1 for _,a in G.nodes(data=True) if a.get("node_type")==NODE_DRUG),
        "cancer_nodes":       sum(1 for _,a in G.nodes(data=True) if a.get("node_type")==NODE_CANCER),
        "alteration_nodes":   sum(1 for _,a in G.nodes(data=True) if a.get("node_type")==NODE_ALTERATION),
        "evidence_nodes":     sum(1 for _,a in G.nodes(data=True) if a.get("node_type")==NODE_EVIDENCE),
        "publication_nodes":  sum(1 for _,a in G.nodes(data=True) if a.get("node_type")==NODE_PUBLICATION),
    }
    return stats
