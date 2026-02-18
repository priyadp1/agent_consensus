import ast


def valid_question(example, key="options"):
    # key: the dict field holding the answer options ("options" for GlobalOpinionsQA,
    #      "perspectives" for OpinionsQA)
    options = example.get(key)

    if not options:
        return False

    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
            example[key] = options
        except Exception:
            return False

    if not isinstance(options, list):
        return False

    return 2 <= len(options) <= 5
