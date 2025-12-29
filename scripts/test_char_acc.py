import difflib

def calculate_char_accuracy(preds, targets):
    total_chars = 0
    correct_chars = 0
    for pred, target in zip(preds, targets):
        matcher = difflib.SequenceMatcher(None, pred, target)
        correct_chars += sum(block.size for block in matcher.get_matching_blocks())
        total_chars += len(target)
    return correct_chars, total_chars

preds = ["29A1234", "29A12345", ""]
targets = ["29A12345", "29A12345", "29A12345"]

correct, total = calculate_char_accuracy(preds, targets)
print(f"Correct: {correct}, Total: {total}, Acc: {correct/total:.4f}")

# Expected:
# 1. "29A1234" vs "29A12345": 7 matches
# 2. "29A12345" vs "29A12345": 8 matches
# 3. "" vs "29A12345": 0 matches
# Total correct: 15
# Total chars: 8 + 8 + 8 = 24
# Acc: 15/24 = 0.625
