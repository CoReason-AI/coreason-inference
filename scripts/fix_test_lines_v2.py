import os
import re

TEST_DIR = "tests"

def update_test_lines():
    for root, _, files in os.walk(TEST_DIR):
        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()

            new_lines = []
            modified = False

            for line in lines:
                if len(line) > 120 and ("estimate_effect" in line or "analyze" in line or "induce_rules" in line or "run_virtual_trial" in line):
                    # Check if context argument exists
                    if "context=" in line:
                        # Split by comma to find where context argument starts
                        parts = line.split(",")

                        # Find the part with context=
                        context_idx = -1
                        for i, p in enumerate(parts):
                            if "context=" in p:
                                context_idx = i
                                break

                        if context_idx != -1:
                            # Reconstruct line breaking before context
                            # Get indentation
                            indent = len(line) - len(line.lstrip())

                            # Join parts before context
                            pre_context = ",".join(parts[:context_idx]) + ","
                            # The context part and everything after
                            post_context = ",".join(parts[context_idx:])

                            # Create new lines
                            new_line_1 = pre_context + "\n"
                            new_line_2 = " " * (indent + 4) + post_context.lstrip()

                            new_lines.append(new_line_1)
                            new_lines.append(new_line_2)
                            modified = True
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            if modified:
                with open(filepath, "w") as f:
                    f.writelines(new_lines)
                print(f"Updated line breaks in {filename}")

if __name__ == "__main__":
    update_test_lines()
