#!/usr/bin/env python3
"""
copy_joint_ranges.py

Parse the joint range attributes from a source MuJoCo XML and
overwrite the ranges of the same-named joints in a target XML.

Usage:
    python copy_joint_ranges.py \\
        --source s110_intermimic.xml \\
        --target s15.xml \\
        --output s15_intermimic.xml
"""

import argparse
import os
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import os

def parse_joint_ranges(xml_path: str) -> Dict[str, str]:
    """
    Extract every joint name and its range attribute from an XML file.
    Args:
        xml_path: path to the XML file
    Returns:
        { "L_Hip_x": "-180.00 180.00", "L_Index1_z": "-55.6250 55.6250", ... }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML ({xml_path}): {e}")
        sys.exit(1)

    joint_ranges = {}
    for joint in root.iter('joint'):
        # freejoints have no range attribute, so they are skipped
        name = joint.get('name')
        range_val = joint.get('range')

        if range_val is not None:  # only joints with a range
            joint_ranges[name] = range_val

    return joint_ranges


def get_joint_names(xml_path: str) -> List[str]:
    """
    Extract joint names from an XML file in order (used for validation).
    Freejoints are excluded.

    Args:
        xml_path: path to the XML file

    Returns:
        list of joint names (order preserved)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML ({xml_path}): {e}")
        sys.exit(1)

    joint_names = []
    for joint in root.iter('joint'):
        # only joints with a range (excludes freejoints)
        if joint.get('range') is not None:
            joint_names.append(joint.get('name'))

    return joint_names


def validate_joint_structure(source_names: List[str], target_names: List[str], force: bool = False) -> None:
    """
    Check that the two files have the same set of joint names.

    Args:
        source_names: joint names in the source file
        target_names: joint names in the target file
        force: if True, ignore warnings and continue
    """
    source_set = set(source_names)
    target_set = set(target_names)

    print(f"[INFO] Joint count: source={len(source_names)}, target={len(target_names)}")

    # Check the count
    if len(source_names) != len(target_names):
        print(f"[WARNING] Joint count mismatch")
        if not force:
            sys.exit(1)

    # Check the name sets
    if source_set != target_set:
        only_in_source = source_set - target_set
        only_in_target = target_set - source_set

        if only_in_source:
            print(f"[WARNING] Joints only in source: {sorted(only_in_source)}")
        if only_in_target:
            print(f"[WARNING] Joints only in target: {sorted(only_in_target)}")

        if not force:
            sys.exit(1)

    # Check the order
    if source_names != target_names:
        print(f"[WARNING] Joint order differs (continuing anyway)")

    print(f"[INFO] Joint structure check: OK")


def replace_ranges_in_text(
    target_text: str,
    source_ranges: Dict[str, str],
    verbose: bool = False
) -> Tuple[str, int, List[str]]:
    """
    Replace each joint's range attribute in the target XML text with the source value.

    Args:
        target_text: full text of the target XML
        source_ranges: { joint_name: range_string }
        verbose: whether to print each replacement

    Returns:
        (modified text, number of replaced joints, list of joints that could not be replaced)
    """
    lines = target_text.split('\n')
    replaced_count = 0
    missed = []

    for joint_name, new_range in source_ranges.items():
        found = False

        for i, line in enumerate(lines):
            # Match the joint tag line with this name and type="hinge"
            if f'name="{joint_name}"' in line and 'type="hinge"' in line:
                old_line = line
                # Replace only the range attribute
                new_line = re.sub(
                    r'range="[^"]*"',
                    f'range="{new_range}"',
                    line
                )

                found = True
                if new_line != old_line:
                    lines[i] = new_line
                    replaced_count += 1

                    if verbose:
                        print(f"  {joint_name}: {old_line.strip()}")
                        print(f"           ↓")
                        print(f"  {joint_name}: {new_line.strip()}")
                else:
                    if verbose:
                        print(f"  {joint_name}: range already identical")
                break

        if not found:
            missed.append(joint_name)

    return '\n'.join(lines), replaced_count, missed


def post_validate(output_path: str, source_ranges: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Re-parse the saved output file and check that every range is correct.

    Args:
        output_path: path of the output file to check
        source_ranges: expected range values

    Returns:
        list of mismatches [(joint_name, expected, actual), ...]
    """
    output_ranges = parse_joint_ranges(output_path)
    mismatches = []

    for joint_name, expected_range in source_ranges.items():
        actual_range = output_ranges.get(joint_name)
        if actual_range != expected_range:
            mismatches.append((joint_name, expected_range, actual_range))

    return mismatches


def copy_joint_ranges(base_path: str, source_path: str, target_path: str,  output_path: str, dry_run: bool = False,    force: bool = False,    verbose: bool = False,    no_validate: bool = False) -> None:
    """
    Main entry point. Runs the full pipeline.

    Args:
        source_path: path to the source XML file
        target_path: path to the target XML file
        output_path: path to the output file
        dry_run: if True, preview only without saving
        force: if True, ignore warnings and force the run
        verbose: if True, print details
        no_validate: if True, skip post-save validation
    """

    source_path = os.path.join(base_path, source_path)
    target_path = os.path.join(base_path, target_path)
    output_path = os.path.join(base_path, output_path)

    # 2. Parse ranges
    print(f"[INFO] Parsing source: {source_path}")
    source_ranges = parse_joint_ranges(source_path)
    print(f"       Extracted {len(source_ranges)} joint ranges")

    # 3. Check joint structure
    print(f"[INFO] Checking joint structure...")
    source_names = get_joint_names(source_path)
    target_names = get_joint_names(target_path)
    validate_joint_structure(source_names, target_names, force=force)

    # 4. Read the target file text
    print(f"[INFO] Reading target file: {target_path}")
    with open(target_path, 'r', encoding='utf-8') as f:
        target_text = f.read()

    # 5. Replace ranges
    print(f"[INFO] Replacing ranges...")
    modified_text, replaced_count, missed = replace_ranges_in_text(
        target_text,
        source_ranges,
        verbose=verbose
    )

    print(f"[INFO] Replaced {replaced_count}/{len(source_ranges)} joints")

    if missed:
        print(f"[ERROR] Could not replace the range of these joints: {missed}")
        sys.exit(1)

    skipped_count = len(source_ranges) - replaced_count - len(missed)
    if skipped_count > 0:
        print(f"[INFO] {skipped_count} joints already had the same range (no change needed)")

    # 6. Save or preview
    if dry_run:
        print(f"\n[DRY-RUN] Would save to: {output_path}")
        print(f"Remove --dry-run to actually save.")
    else:
        # Confirm before overwriting an existing file
        if Path(output_path).exists() and not force:
            print(f"\n[WARNING] {output_path} already exists.")
            ans = input(f"Overwrite? [y/N]: ").strip().lower()
            if ans != 'y':
                print("[INFO] Cancelled.")
                sys.exit(0)

        print(f"[INFO] Saving: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modified_text)
        print(f"[SUCCESS] File saved")

        # 7. Post-save validation
        if not no_validate:
            print(f"[INFO] Validating output...")
            mismatches = post_validate(output_path, source_ranges)

            if mismatches:
                print(f"[ERROR] Range mismatch for {len(mismatches)} joints:")
                for joint_name, expected, actual in mismatches:
                    print(f"       {joint_name}: expected={expected}, actual={actual}")

                # Delete the file so a bad output is not left behind
                Path(output_path).unlink()
                print(f"[ERROR] Output file deleted.")
                sys.exit(1)
            else:
                print(f"[SUCCESS] All {len(source_ranges)} joint ranges match")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Copy joint ranges from a source MuJoCo XML to a target XML."
    )
    parser.add_argument("--source", "-s", type=str, default="s110_intermimic_ROM.xml", help="Source XML file to copy ranges from (default: %(default)s)"    )
    parser.add_argument("--target", "-t", type=str, default="s94.xml", help="Target XML file to receive the ranges (default: %(default)s)"    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path. Defaults to the target file name with an _intermimic suffix"    )
    parser.add_argument("--dry-run", action="store_true", help="Print the changes without saving")
    parser.add_argument("--force", "-f", action="store_true", help="Ignore joint structure mismatch warnings and allow overwriting an existing file")
    parser.add_argument("--verbose", "-v",action="store_true",help="Print each joint's range change in detail"    )
    parser.add_argument("--no-validate",action="store_true",help="Skip post-save validation (re-parsing check)"    )
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        target_p = Path(args.target)
        args.output = str(target_p.parent / (target_p.stem + "_intermimic" + target_p.suffix))
    base_path = "wristmimic/data/assets/parahome/sim_human/"
    copy_joint_ranges(
        base_path=base_path,
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
        no_validate=args.no_validate
    )


if __name__ == "__main__":
    import sys
    # sys.argv = ['wristmimic/data/assets/parahome/sim_human/copy_joint_ranges.py', 
    #             "--source", "s110_intermimic.xml"
    #             "--target", "s15.xml"
    #             ]  
    main()
