"""
cli_panelbuild.py — additions to mindv CLI for `mindv panel-build`

Merge this block into cli.py:
  1. Add cmd_panel_build() function
  2. In build_parser(), add subparser "panel-build"

Kept as a separate file so it can be reviewed / merged cleanly.
"""

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def cmd_panel_build(args):
    """
    Entry point for `mindv panel-build`.

    Validates arguments, sets up logging, delegates to panel_builder.run_panel_build().
    """
    from .panel_builder import run_panel_build

    # Derive reference accession from --ref if not supplied
    ref_acc = getattr(args, "ref_accession", None) or ""

    # Genes list
    genes = [g.strip() for g in args.genes.split(",")] if args.genes else None

    log.info("mindv panel-build starting")
    log.info("  Organism  : %s", args.organism)
    log.info("  Reference : %s", args.ref)
    log.info("  Source    : %s", args.source)
    log.info("  GFF3      : %s", args.gff)
    log.info("  Output    : %s", args.output)
    if genes:
        log.info("  Genes     : %s", ", ".join(genes))

    try:
        panel = run_panel_build(
            source=args.source,
            organism=args.organism,
            ref_fasta=args.ref,
            gff_path=args.gff,
            output_path=args.output,
            card_json=getattr(args, "card_json", None),
            amrfinder_tsv=getattr(args, "amrfinder_tsv", None),
            who_catalogue_tsv=getattr(args, "who_catalogue_tsv", None),
            vcf_path=getattr(args, "vcf_path", None),
            vcf_annotation_tsv=getattr(args, "vcf_annotation_tsv", None),
            genes=genes,
            who_min_grade=getattr(args, "who_min_grade", 2),
            panel_version=getattr(args, "panel_version", "1.0"),
            reference_accession=ref_acc,
            fail_on_error=getattr(args, "strict", False),
        )
    except Exception as exc:
        log.error("panel-build failed: %s", exc)
        sys.exit(1)

    # Print summary to stdout
    stats = panel.get("panel_build_stats", {})
    n_targets = len(panel.get("targets", {}))
    total_entries = sum(
        len(t.get("known_mutations", {}))
        for t in panel.get("targets", {}).values()
    )
    print(f"\n{'='*60}")
    print(f"  mindv panel-build complete")
    print(f"  Organism  : {panel['organism']}")
    print(f"  Reference : {panel['reference']}")
    print(f"  Genes     : {n_targets}")
    print(f"  Entries   : {total_entries} (verified: {stats.get('verified',0)}, "
          f"failed: {stats.get('failed',0)})")
    print(f"  Output    : {args.output}")
    print(f"{'='*60}\n")

    if stats.get("failed", 0) > 0:
        print(
            f"WARNING: {stats['failed']} mutations failed verification. "
            "Review the JSON 'verification' fields and check gene coordinates.\n"
        )


def add_panel_build_subparser(subparsers: argparse._SubParsersAction):
    """
    Add the `panel-build` subparser to the mindv CLI.

    Call this from build_parser() in cli.py:
        from .cli_panelbuild import add_panel_build_subparser
        add_panel_build_subparser(sub)
    """
    p = subparsers.add_parser(
        "panel-build",
        help="Build and verify an AMR panel JSON from CARD / AMRFinderPlus / WHO catalogue",
        description="""
mindv panel-build — Automated, computationally verified AMR panel generator.

Takes a reference genome + resistance mutation database and produces a
verified mindv panel JSON. Every entry is verified by extracting the
reference codon directly from the FASTA and confirming it translates to
the claimed reference amino acid. Entries that fail verification are
reported and excluded from the output.

Examples
--------
# Build TB panel from WHO catalogue (grades 1-2 only)
mindv panel-build \\
    --source who_catalogue \\
    --who-catalogue-tsv WHO_catalogue_v2_MTBC_2023.tsv \\
    --organism "Mycobacterium tuberculosis" \\
    --ref NC_000962.3.fasta \\
    --gff NC_000962.3.gff3 \\
    --genes rpoB,katG,gyrA,gyrB,embB,pncA,rpsL,rrs,inhA,eis \\
    --output configs/tb_h37rv.json

# Build from CARD database
mindv panel-build \\
    --source card \\
    --card-json card.json \\
    --organism "Neisseria gonorrhoeae" \\
    --ref GCF_000006845.1.fasta \\
    --gff GCF_000006845.1.gff3 \\
    --output configs/ngon.json

# Combine CARD + WHO catalogue sources
mindv panel-build \\
    --source combined \\
    --card-json card.json \\
    --who-catalogue-tsv WHO_catalogue.tsv \\
    --organism "Mycobacterium tuberculosis" \\
    --ref NC_000962.3.fasta \\
    --gff NC_000962.3.gff3 \\
    --output configs/tb_combined.json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument(
        "--source", required=True,
        choices=["card", "amrfinderplus", "who_catalogue", "vcf", "combined"],
        help="Resistance mutation database to use as input.",
    )
    p.add_argument(
        "--organism", required=True,
        help='Species name as it appears in the database (e.g. "Mycobacterium tuberculosis").',
    )
    p.add_argument(
        "--ref", required=True, metavar="FASTA",
        help="Reference genome FASTA (will be indexed with samtools faidx if needed).",
    )
    p.add_argument(
        "--gff", required=True, metavar="GFF3",
        help="NCBI GFF3 annotation file for the reference genome (.gz accepted).",
    )
    p.add_argument(
        "--output", required=True, metavar="JSON",
        help="Output path for the verified panel JSON.",
    )

    # Source-specific
    src_group = p.add_argument_group("Source database files")
    src_group.add_argument(
        "--card-json", metavar="FILE",
        help="Path to CARD card.json (required when --source card or combined).",
    )
    src_group.add_argument(
        "--amrfinder-tsv", metavar="FILE",
        help="Path to AMRFinderPlus mutation TSV (required when --source amrfinderplus or combined).",
    )
    src_group.add_argument(
        "--who-catalogue-tsv", metavar="FILE",
        help="Path to WHO mutation catalogue TSV (required when --source who_catalogue or combined).",
    )
    src_group.add_argument(
        "--vcf", metavar="FILE", dest="vcf_path",
        help="VCF file of known resistance variants (required when --source vcf). "
             "INFO fields GENE=, DRUG=, AA_CHANGE= or SnpEff ANN= are used for annotation.",
    )
    src_group.add_argument(
        "--vcf-annotation-tsv", metavar="FILE", dest="vcf_annotation_tsv",
        help="Optional companion TSV (pos, ref, alt, gene, drug, aa_change) to annotate VCF records "
             "that lack INFO GENE/DRUG fields.",
    )

    # Optional
    opt_group = p.add_argument_group("Optional parameters")
    opt_group.add_argument(
        "--genes", metavar="GENE_LIST",
        help="Comma-separated list of gene names to include (default: all genes in database). "
             "Example: --genes rpoB,katG,gyrA,gyrB",
    )
    opt_group.add_argument(
        "--who-min-grade", type=int, default=2, metavar="N",
        help="Maximum WHO confidence grade to include (1=definitely associated, 3=uncertain). "
             "Default: 2 (grades 1 and 2 only).",
    )
    opt_group.add_argument(
        "--ref-accession", metavar="ACC",
        help="Reference genome accession for panel metadata (e.g. NC_000962.3). "
             "Inferred from --ref filename if not provided.",
    )
    opt_group.add_argument(
        "--panel-version", metavar="VER", default="1.0",
        help="Version string embedded in panel JSON metadata. Default: 1.0.",
    )
    opt_group.add_argument(
        "--strict", action="store_true",
        help="Abort immediately if any entry fails verification (default: skip and warn).",
    )

    p.set_defaults(func=cmd_panel_build)
    return p
