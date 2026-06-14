"""Fix JSONL files: replace unescaped inner quotes with Chinese brackets."""

import json
import re
from pathlib import Path


def fix_jsonl_file(filepath: str) -> int:
    """Read a JSONL file, fix unescaped quotes, rewrite.
    
    Returns number of lines fixed.
    """
    path = Path(filepath)
    with open(path, "rb") as f:
        content = f.read().decode("utf-8")
    
    lines = content.split("\n")
    fixed_count = 0
    fixed_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            fixed_lines.append("")
            continue
        
        # Try parsing first
        try:
            obj = json.loads(line)
            fixed_lines.append(json.dumps(obj, ensure_ascii=False))
            continue
        except json.JSONDecodeError:
            pass
        
        # Try to fix: replace Chinese-style quoted text patterns
        # Pattern: Chinese text between ASCII quotes inside a JSON string value
        # Strategy: find pairs of " inside string values and replace with 
        # Chinese brackets
        
        # Simple approach: replace "X" patterns (where X is Chinese) with X
        # This regex matches " followed by 1-20 non-quote chars followed by "
        # but only when preceded by a Chinese char or followed by a Chinese char
        
        # More robust: manually parse and fix
        fixed_line = _fix_inner_quotes(line)
        
        try:
            obj = json.loads(fixed_line)
            fixed_lines.append(json.dumps(obj, ensure_ascii=False))
            fixed_count += 1
        except json.JSONDecodeError as e:
            print(f"  Line {i+1}: Still broken after fix attempt: {e}")
            print(f"    {fixed_line[:100]}...")
            fixed_lines.append(line)  # Keep original
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines))
    
    return fixed_count


def _fix_inner_quotes(line: str) -> str:
    """Fix unescaped double quotes inside JSON string values.
    
    Strategy: Parse character by character, tracking JSON structure.
    When inside a string value and encountering an unescaped quote
    that isn't the closing quote, replace it.
    """
    result = []
    i = 0
    in_string = False
    string_start = -1
    
    while i < len(line):
        ch = line[i]
        
        if ch == "\\" and in_string and i + 1 < len(line):
            # Escaped character - copy both
            result.append(ch)
            result.append(line[i + 1])
            i += 2
            continue
        
        if ch == '"':
            if not in_string:
                in_string = True
                string_start = i
                result.append(ch)
            else:
                # Could be closing quote or inner quote
                # Look ahead to determine
                rest = line[i + 1:].lstrip()
                if rest and rest[0] in ":,}]":
                    # This is a closing quote (followed by JSON delimiter)
                    in_string = False
                    result.append(ch)
                elif not rest:
                    # End of line
                    in_string = False
                    result.append(ch)
                else:
                    # This is an inner quote - replace with Chinese bracket
                    # Find the matching closing inner quote
                    j = i + 1
                    while j < len(line) and line[j] != '"':
                        j += 1
                    if j < len(line):
                        # Replace both inner quotes with Chinese brackets
                        inner_text = line[i + 1:j]
                        result.append("\u300c")  # 「
                        result.append(inner_text)
                        result.append("\u300d")  # 」
                        i = j + 1
                        continue
                    else:
                        # No matching quote found, treat as closing
                        in_string = False
                        result.append(ch)
        else:
            result.append(ch)
        
        i += 1
    
    return "".join(result)


def main():
    files = [
        "datasets/seed/customer_service.jsonl",
        "datasets/seed/rag_evaluation.jsonl",
        "datasets/seed/safety_redteam.jsonl",
    ]
    
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"[SKIP] {filepath} not found")
            continue
        
        print(f"Processing {filepath}...")
        fixed = fix_jsonl_file(filepath)
        
        # Verify
        with open(path, "rb") as f:
            content = f.read().decode("utf-8")
        lines = [l for l in content.split("\n") if l.strip()]
        ok = 0
        for line in lines:
            try:
                json.loads(line.strip())
                ok += 1
            except json.JSONDecodeError:
                pass
        
        print(f"  Fixed: {fixed} lines, Verified: {ok}/{len(lines)} lines parse OK")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
