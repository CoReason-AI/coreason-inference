import os
import re

TEST_DIR = "tests"

def update_test_signatures():
    for root, _, files in os.walk(TEST_DIR):
        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)
            with open(filepath, "r") as f:
                content = f.read()

            # Add -> None if missing
            # Logic: Match `def test_name(...)` that is NOT followed by `->`.
            # Note: This is fragile with regex but sufficient for standard test files.
            # We look for `):` and replace with `) -> None:`

            # This regex looks for `def test_...):` where `)` is the closing parenthesis of args.
            # It avoids replacing if `->` is already present before `:`.

            # Explanation:
            # `def test_`: start
            # `[^:]+`: arguments (non-greedy would be better but we rely on no colon inside args usually)
            # `\)`: closing paren
            # `\s*:`: optional space and colon
            # Negative lookbehind `(?<!->)` doesn't work well if there are spaces.

            # Better approach: Iterate lines/chunks.

            new_content = re.sub(r"(def test_[a-zA-Z0-9_]+\s*\(.*?\))(\s*:)", r"\1 -> None\2", content)

            # Also need to fix imports if missing? No, usually imports are fine.

            if new_content != content:
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"Added return types to {filename}")

if __name__ == "__main__":
    update_test_signatures()
