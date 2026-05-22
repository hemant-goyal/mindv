"""
who_resistance_classifier.py
=============================
WHO 2022 resistance classification for M. tuberculosis.

WHO 2022 updated definitions (WHO consolidated guidelines, Module 4, 2022):
  Susceptible    : No resistance detected
  Mono-resistant : Resistant to exactly 1 first-line drug
  Poly-resistant : Resistant to ≥2 first-line drugs, NOT meeting MDR criteria
  RR-TB          : Rifampicin resistant (INH status unknown/susceptible)
  MDR-TB         : Resistant to Rifampicin AND Isoniazid
  Pre-XDR-TB     : MDR + resistant to any Fluoroquinolone
  XDR-TB         : Pre-XDR + resistant to ≥1 of Bedaquiline OR Linezolid

NOTE: WHO 2022 changed the XDR definition. The OLD (pre-2021) definition
      used second-line injectables (AMK/KAN/CAP) for XDR. The NEW definition
      uses bedaquiline or linezolid. Amikacin resistance now classifies as
      Pre-XDR (if FQ+) or is reported separately, not XDR.

      Since current panels may not include BDQ/LZD targets, samples with
      MDR+FQ+AMK are classified as Pre-XDR (not XDR) under the 2022 definition,
      with a note flagging amikacin resistance for extended DST.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from typing import Set, Dict

DRUG_TO_GROUP = {
    "Rifampicin":                       "Rifampicin",
    "Rifampicin/Rifapentine":           "Rifampicin",
    "Rifapentine":                      "Rifampicin",
    "Isoniazid":                        "Isoniazid",
    "Ethionamide":                      "Ethionamide",
    "Fluoroquinolones":                 "Fluoroquinolones",
    "Levofloxacin":                     "Fluoroquinolones",
    "Moxifloxacin":                     "Fluoroquinolones",
    "Ofloxacin":                        "Fluoroquinolones",
    "Amikacin":                         "Amikacin",
    "Amikacin/Kanamycin/Capreomycin":   "Amikacin",
    "Kanamycin":                        "Amikacin",
    "Capreomycin":                      "Amikacin",
    "Ethambutol":                       "Ethambutol",
    "Streptomycin":                     "Streptomycin",
    "Pyrazinamide":                     "Pyrazinamide",
    "Bedaquiline":                      "Bedaquiline",
    "Linezolid":                        "Linezolid",
}

FIRST_LINE = {"Rifampicin", "Isoniazid", "Pyrazinamide", "Ethambutol"}


def classify_sample(resistant_drugs: Set[str]) -> Dict:
    r = resistant_drugs
    rif = "Rifampicin"       in r
    inh = "Isoniazid"        in r
    fq  = "Fluoroquinolones" in r
    amk = "Amikacin"         in r
    bdq = "Bedaquiline"      in r
    lzd = "Linezolid"        in r
    n_fl = len(r & FIRST_LINE)

    if not r:
        return {"classification": "Susceptible",
                "resistant_drugs": [],
                "notes": "No resistance-associated mutations detected in panel targets."}

    # XDR (WHO 2022): Pre-XDR + BDQ or LZD
    if rif and inh and fq and (bdq or lzd):
        return {"classification": "XDR-TB",
                "resistant_drugs": sorted(r),
                "notes": "Extensively drug-resistant TB (WHO 2022): MDR + Fluoroquinolone + "
                         "Bedaquiline/Linezolid resistance. Very limited treatment options."}

    # Pre-XDR: MDR + FQ (AMK resistance noted but does not change WHO 2022 category)
    if rif and inh and fq:
        note = ("Pre-extensively drug-resistant TB (WHO 2022): MDR + Fluoroquinolone resistance. "
                "Recommend bedaquiline/linezolid DST for XDR classification.")
        if amk:
            note += " Amikacin resistance also detected (extend DST panel)."
        return {"classification": "Pre-XDR-TB",
                "resistant_drugs": sorted(r),
                "notes": note}

    # MDR: RIF + INH
    if rif and inh:
        note = "Multidrug-resistant TB: Rifampicin + Isoniazid resistant. BPaLM regimen recommended."
        if amk:
            note += " Amikacin resistance also detected; consider FQ DST for Pre-XDR classification."
        return {"classification": "MDR-TB",
                "resistant_drugs": sorted(r),
                "notes": note}

    # RR-TB: RIF only (no confirmed INH)
    if rif and not inh:
        return {"classification": "RR-TB",
                "resistant_drugs": sorted(r),
                "notes": "Rifampicin-resistant TB. Treat as MDR until INH DST confirmed. BPaLM recommended."}

    # INH mono (common, important)
    if inh and not rif:
        if n_fl == 1 and len(r) == 1:
            return {"classification": "Mono-resistant (Isoniazid)",
                    "resistant_drugs": sorted(r),
                    "notes": "INH mono-resistant TB. WHO recommends RZES x6 months."}
        return {"classification": "Poly-resistant",
                "resistant_drugs": sorted(r),
                "notes": "Poly-resistant TB: ≥2 drugs resistant, not MDR criteria."}

    # Poly-resistant: ≥2 first-line, not MDR
    if n_fl >= 2:
        return {"classification": "Poly-resistant",
                "resistant_drugs": sorted(r),
                "notes": "Poly-resistant TB: ≥2 first-line drugs resistant, not MDR criteria."}

    # Mono-resistant (single drug)
    if len(r) == 1:
        drug = next(iter(r))
        return {"classification": f"Mono-resistant ({drug})",
                "resistant_drugs": sorted(r),
                "notes": f"Mono-resistant to {drug}. Standard regimen with substitution."}

    return {"classification": "Other-resistant",
            "resistant_drugs": sorted(r),
            "notes": f"Resistance to: {', '.join(sorted(r))}. Clinical review recommended."}


def classify_cohort(mindv_csv: str) -> Dict[str, Dict]:
    sample_drugs: Dict[str, Set[str]] = defaultdict(set)
    with open(mindv_csv) as f:
        for row in csv.DictReader(f):
            if row.get("filter") != "PASS" or row.get("tier") != "KNOWN":
                continue
            group = DRUG_TO_GROUP.get(row.get("known_drug", "").strip())
            if group:
                sample_drugs[row["patient"]].add(group)
    return {s: classify_sample(d) for s, d in sample_drugs.items()}


def write_cohort_classification(mindv_csv: str, out_tsv: str) -> None:
    classifications = classify_cohort(mindv_csv)
    # Add susceptible samples
    with open(mindv_csv) as f:
        all_samples = {row["patient"] for row in csv.DictReader(f)}
    for s in all_samples:
        if s not in classifications:
            classifications[s] = classify_sample(set())

    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sample", "who_classification", "resistant_drugs", "notes"])
        for sample in sorted(classifications):
            r = classifications[sample]
            w.writerow([sample,
                        r["classification"],
                        "; ".join(r["resistant_drugs"]) if r["resistant_drugs"] else "—",
                        r["notes"]])
    print(f"Classification written: {out_tsv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python who_resistance_classifier.py cohort_mindv.csv [out.tsv]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "cohort_who_classification.tsv"
    write_cohort_classification(csv_path, out_path)

    cls = classify_cohort(csv_path)
    print()
    print(f"  {'Sample':15s}  {'Classification':30s}  Resistant drugs")
    print(f"  {'-'*15}  {'-'*30}  {'-'*35}")
    for s in sorted(cls):
        r = cls[s]
        drugs = ", ".join(r["resistant_drugs"]) if r["resistant_drugs"] else "—"
        print(f"  {s:15s}  {r['classification']:30s}  {drugs}")
