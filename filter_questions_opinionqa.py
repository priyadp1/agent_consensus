import ast
def valid_question(example):
    options = example.get("perspectives")

    if not options:
        return False

    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
            example["perspectives"] = options
        except Exception:
            return False

    if not isinstance(options, list):
        return False

    return 2 <= len(options) <= 5