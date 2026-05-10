#!/usr/bin/env python3
"""Update version to 0.4.1"""

# Update pyproject.toml
with open('pyproject.toml', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('version = "0.4.0"', 'version = "0.4.1"')

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated pyproject.toml to version 0.4.1")

# Verify
with open('pyproject.toml', 'r', encoding='utf-8') as f:
    for line in f:
        if 'version' in line and '=' in line:
            print(f"   {line.strip()}")
