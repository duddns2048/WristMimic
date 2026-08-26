#!/usr/bin/env python3
"""
copy_joint_ranges.py

소스 MuJoCo XML에서 joint range 속성을 파싱하여,
타겟 XML의 동일 joint name에 덮어씁니다.

사용법:
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
    XML 파일에서 모든 joint 이름과 range 속성값을 추출한다.
    Args:
        xml_path: XML 파일 경로
    Returns:
        { "L_Hip_x": "-180.00 180.00", "L_Index1_z": "-55.6250 55.6250", ... }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML 파싱 실패 ({xml_path}): {e}")
        sys.exit(1)

    joint_ranges = {}
    for joint in root.iter('joint'):
        # freejoint는 range 속성이 없으므로 skip
        name = joint.get('name')
        range_val = joint.get('range')

        if range_val is not None:  # range가 있는 것만
            joint_ranges[name] = range_val

    return joint_ranges


def get_joint_names(xml_path: str) -> List[str]:
    """
    XML 파일에서 joint 이름 목록을 순서대로 추출한다. (검증용)
    freejoint 제외.

    Args:
        xml_path: XML 파일 경로

    Returns:
        joint name 리스트 (순서 보존)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[ERROR] XML 파싱 실패 ({xml_path}): {e}")
        sys.exit(1)

    joint_names = []
    for joint in root.iter('joint'):
        # range가 있는 것만 (freejoint 제외)
        if joint.get('range') is not None:
            joint_names.append(joint.get('name'))

    return joint_names


def validate_joint_structure(source_names: List[str], target_names: List[str], force: bool = False) -> None:
    """
    두 파일의 joint 이름 목록이 동일한지 검증한다.

    Args:
        source_names: 소스 파일의 joint 이름 리스트
        target_names: 타겟 파일의 joint 이름 리스트
        force: True면 경고 무시하고 계속 진행
    """
    source_set = set(source_names)
    target_set = set(target_names)

    print(f"[INFO] Joint 수: 소스={len(source_names)}, 타겟={len(target_names)}")

    # 개수 확인
    if len(source_names) != len(target_names):
        print(f"[WARNING] joint 수 불일치")
        if not force:
            sys.exit(1)

    # 이름 집합 확인
    if source_set != target_set:
        only_in_source = source_set - target_set
        only_in_target = target_set - source_set

        if only_in_source:
            print(f"[WARNING] 소스에만 있는 joint: {sorted(only_in_source)}")
        if only_in_target:
            print(f"[WARNING] 타겟에만 있는 joint: {sorted(only_in_target)}")

        if not force:
            sys.exit(1)

    # 순서 확인
    if source_names != target_names:
        print(f"[WARNING] joint 순서가 다릅니다 (그러나 계속 진행)")

    print(f"[INFO] Joint 구조 검증: OK")


def replace_ranges_in_text(
    target_text: str,
    source_ranges: Dict[str, str],
    verbose: bool = False
) -> Tuple[str, int, List[str]]:
    """
    타겟 XML 텍스트에서 각 joint의 range 속성값을 소스 값으로 교체한다.

    Args:
        target_text: 타겟 XML 전체 텍스트
        source_ranges: { joint_name: range_string }
        verbose: 각 치환 내역을 출력할지 여부

    Returns:
        (수정된 텍스트, 치환된 joint 수, 치환 실패한 joint 목록)
    """
    lines = target_text.split('\n')
    replaced_count = 0
    missed = []

    for joint_name, new_range in source_ranges.items():
        found = False

        for i, line in enumerate(lines):
            # 해당 joint name이 포함된 joint 태그 라인이고 type="hinge"인지 확인
            if f'name="{joint_name}"' in line and 'type="hinge"' in line:
                old_line = line
                # range 속성만 교체
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
                        print(f"  {joint_name}: 이미 동일한 range")
                break

        if not found:
            missed.append(joint_name)

    return '\n'.join(lines), replaced_count, missed


def post_validate(output_path: str, source_ranges: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    저장된 출력 파일을 재파싱하여 모든 range가 올바른지 검증한다.

    Args:
        output_path: 검증할 출력 파일 경로
        source_ranges: 기대되는 range 값들

    Returns:
        불일치 항목 리스트 [(joint_name, expected, actual), ...]
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
    메인 실행 함수. 전체 파이프라인을 실행한다.

    Args:
        source_path: 소스 XML 파일 경로
        target_path: 타겟 XML 파일 경로
        output_path: 출력 파일 경로
        dry_run: True면 저장 없이 미리보기만 함
        force: True면 경고 무시하고 강제 실행
        verbose: True면 상세 출력
        no_validate: True면 사후 검증 건너뜀
    """

    source_path = os.path.join(base_path, source_path)
    target_path = os.path.join(base_path, target_path)
    output_path = os.path.join(base_path, output_path)

    # 2. Range 파싱
    print(f"[INFO] 소스 파싱: {source_path}")
    source_ranges = parse_joint_ranges(source_path)
    print(f"       {len(source_ranges)}개 joint range 추출 완료")

    # 3. Joint 구조 검증
    print(f"[INFO] Joint 구조 검증...")
    source_names = get_joint_names(source_path)
    target_names = get_joint_names(target_path)
    validate_joint_structure(source_names, target_names, force=force)

    # 4. 타겟 파일 텍스트 읽기
    print(f"[INFO] 타겟 파일 읽기: {target_path}")
    with open(target_path, 'r', encoding='utf-8') as f:
        target_text = f.read()

    # 5. Range 치환
    print(f"[INFO] Range 치환 중...")
    modified_text, replaced_count, missed = replace_ranges_in_text(
        target_text,
        source_ranges,
        verbose=verbose
    )

    print(f"[INFO] 치환 완료: {replaced_count}/{len(source_ranges)}개 joint")

    if missed:
        print(f"[ERROR] 다음 joint의 range를 치환하지 못했습니다: {missed}")
        sys.exit(1)

    skipped_count = len(source_ranges) - replaced_count - len(missed)
    if skipped_count > 0:
        print(f"[INFO] {skipped_count}개 joint는 이미 동일한 range (변경 불필요)")

    # 6. 저장 또는 미리보기
    if dry_run:
        print(f"\n[DRY-RUN] 다음 위치에 저장될 예정입니다: {output_path}")
        print(f"실제로 저장하려면 --dry-run 옵션을 제거하세요.")
    else:
        # 기존 파일 덮어쓰기 확인
        if Path(output_path).exists() and not force:
            print(f"\n[WARNING] {output_path} 이미 존재합니다.")
            ans = input(f"덮어쓸까요? [y/N]: ").strip().lower()
            if ans != 'y':
                print("[INFO] 취소되었습니다.")
                sys.exit(0)

        print(f"[INFO] 저장: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modified_text)
        print(f"[SUCCESS] 파일 저장 완료")

        # 7. 사후 검증
        if not no_validate:
            print(f"[INFO] 사후 검증 중...")
            mismatches = post_validate(output_path, source_ranges)

            if mismatches:
                print(f"[ERROR] {len(mismatches)}개 joint의 range가 일치하지 않습니다:")
                for joint_name, expected, actual in mismatches:
                    print(f"       {joint_name}: expected={expected}, actual={actual}")

                # 파일 삭제 (오염 방지)
                Path(output_path).unlink()
                print(f"[ERROR] 출력 파일이 삭제되었습니다.")
                sys.exit(1)
            else:
                print(f"[SUCCESS] 모든 {len(source_ranges)}개 joint range 일치 확인")


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(
        description="소스 MuJoCo XML의 joint range를 타겟 XML에 복사합니다."
    )
    parser.add_argument("--source", "-s", type=str, default="s110_intermimic_ROM.xml", help="범위를 복사해올 소스 XML 파일 경로 (기본값: s110_intermimic.xml)"    )
    parser.add_argument("--target", "-t", type=str, default="s94.xml", help="범위를 받을 타겟 XML 파일 경로 (기본값: s15.xml)"    )
    parser.add_argument("--output", "-o", type=str, default=None, help="출력 파일 경로. 미지정 시 타겟 파일명에 _intermimic 접미사를 붙임"    )
    parser.add_argument("--dry-run", action="store_true", help="실제 저장 없이 변경될 내용만 출력")
    parser.add_argument("--force", "-f", action="store_true", help="joint 구조 불일치 경고 무시 및 기존 파일 덮어쓰기 허용")
    parser.add_argument("--verbose", "-v",action="store_true",help="각 joint의 range 변경 내역을 상세 출력"    )
    parser.add_argument("--no-validate",action="store_true",help="사후 검증(재파싱 확인) 건너뜀"    )
    args = parser.parse_args()

    # output 기본값 처리
    if args.output is None:
        target_p = Path(args.target)
        args.output = str(target_p.parent / (target_p.stem + "_intermimic" + target_p.suffix))
    base_path = "intermimic/data/assets/parahome/sim_human/"
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
    # sys.argv = ['intermimic/data/assets/parahome/sim_human/copy_joint_ranges.py', 
    #             "--source", "s110_intermimic.xml"
    #             "--target", "s15.xml"
    #             ]  
    main()
