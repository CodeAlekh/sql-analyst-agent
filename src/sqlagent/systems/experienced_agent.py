"""Single agent plus experience from the iteration split: the lessons it wrote about its own
mistakes (shown on every question), its wrong vs correct SQL for past mistakes on the same
database and, optionally, solved examples from the same database (conventions). A question
never sees its own entry (leave-one-out on the iteration split)."""

from sqlagent.experience import conventions, load
from sqlagent.systems.single_agent import SingleAgent


def experience_text(mistakes: list[dict], solved: list[dict], question: dict) -> str | None:
    def others(entries, same_db=False):
        return [e for e in entries if e["question_id"] != question["question_id"]
                and (not same_db or e["db_id"] == question["db_id"])]

    lessons, pairs, examples = others(mistakes), others(mistakes, True), others(solved, True)
    if not (lessons or examples):
        return None
    parts = ["Experience from practice questions. Use it to avoid repeating past mistakes and "
             "to match how answers are expected to look."]
    if lessons:
        parts += ["", "Lessons from your past mistakes:"] + [f"- {e['lesson']}" for e in lessons]
    if examples:
        parts += ["", "Solved questions on this database:"]
        for e in examples:
            parts += ["", f"Question: {e['question']}", f"Evidence: {e['evidence'] or '(none)'}",
                      f"SQL:\n{e['gold_sql']}"]
    if pairs:
        parts += ["", "Your past mistakes on this database:"]
        for e in pairs:
            parts += ["", f"Question: {e['question']}", f"Evidence: {e['evidence'] or '(none)'}",
                      f"Your wrong SQL:\n{e['wrong_sql']}", f"Correct SQL:\n{e['gold_sql']}"]
    return "\n".join(parts)


class ExperiencedAgent(SingleAgent):
    name = "experienced_agent"

    def __init__(self, llm, mistakes: list[dict] | None = None, solved: list[dict] | None = None,
                 use_conventions: bool = True, **kwargs):
        super().__init__(llm, **kwargs)
        self.mistakes = load() if mistakes is None else mistakes
        self.solved = [] if not use_conventions else conventions() if solved is None else solved

    def system_prompt(self, question: dict) -> list[dict]:
        blocks = super().system_prompt(question)
        if text := experience_text(self.mistakes, self.solved, question):
            blocks.append({"type": "text", "text": text, "cache_control": {"type": "ephemeral"}})
        return blocks
