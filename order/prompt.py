"""Locked judge prompt. Do not edit the strings: every judge and both orders use them verbatim."""

SYSTEM = (
    "You are an impartial judge. Compare two responses to the user's question. "
    "Judge correctness first, then helpfulness. Do not let response order or length influence you."
)

USER = (
    "[Question]\n{question}\n\n"
    "[Response A]\n{a}\n\n"
    "[Response B]\n{b}\n\n"
    "Output only one token: A, B, or C (tie)."
)

ORDERS = ("AB", "BA")  # AB: answer1 in slot A. BA: answer2 in slot A.


def slot_answers(pair, order):
    """(slot A text, slot B text) for a pair under `order`. `pair` has answer1/answer2."""
    if order == "AB":
        return pair["answer1"], pair["answer2"]
    if order == "BA":
        return pair["answer2"], pair["answer1"]
    raise ValueError(f"order must be AB or BA, got {order!r}")


def build_messages(question, a, b):
    """Chat messages for one judgment (the runner passes them to the judge's chat template)."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(question=question, a=a, b=b)},
    ]
