"""How we score a predicted structure against the ViennaRNA reference."""

import numpy as np
from .rna import dotbracket_to_pairs


def compare_structures(pred_db, ref_db):
    """Base-pair level scores. Returns sensitivity, PPV, F1, MCC-like score."""
    p = set(dotbracket_to_pairs(pred_db))
    r = set(dotbracket_to_pairs(ref_db))
    tp = len(p & r)
    fp = len(p - r)
    fn = len(r - p)

    sens = tp / (tp + fn) if (tp + fn) else 1.0
    ppv = tp / (tp + fp) if (tp + fp) else 1.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
    mcc = np.sqrt(sens * ppv)          # common stand-in for MCC in RNA papers

    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": sens,
            "ppv": ppv, "f1": f1, "mcc": mcc,
            "exact_match": pred_db == ref_db}


def report(seq, pred_db, ref_db, pred_energy, ref_energy):
    """One tidy row of results."""
    row = compare_structures(pred_db, ref_db)
    row.update({
        "length": len(seq),
        "pred_energy": pred_energy,
        "ref_energy": ref_energy,
        "energy_gap": pred_energy - ref_energy,
    })
    return row


def print_row(row):
    print(f"  length {row['length']:3d} | gap {row['energy_gap']:+6.2f} kcal/mol "
          f"| F1 {row['f1']:.2f} | exact {row['exact_match']}")
