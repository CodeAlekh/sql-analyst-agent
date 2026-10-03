from sqlagent.systems.experienced_agent import ExperiencedAgent, experience_text
from sqlagent.systems.single_agent import SingleAgent


def mistake(qid, db_id):
    return {"question_id": qid, "db_id": db_id, "question": f"question {qid}", "evidence": "",
            "wrong_sql": f"SELECT wrong_{qid}", "gold_sql": f"SELECT right_{qid}",
            "lesson": f"lesson {qid}"}


def solved(qid, db_id):
    return {"question_id": qid, "db_id": db_id, "question": f"solved {qid}", "evidence": "",
            "gold_sql": f"SELECT solved_{qid}"}


MISTAKES = [mistake(1, "shop"), mistake(2, "shop"), mistake(3, "zoo")]
SOLVED = [solved(10, "shop"), solved(11, "zoo")]


def test_lessons_from_every_db_pairs_and_examples_only_from_the_same_db():
    text = experience_text(MISTAKES, SOLVED, {"question_id": 99, "db_id": "shop"})
    assert all(f"lesson {i}" in text for i in (1, 2, 3))
    assert "SELECT wrong_1" in text and "SELECT right_2" in text and "SELECT solved_10" in text
    assert "SELECT wrong_3" not in text and "SELECT solved_11" not in text


def test_a_question_never_sees_its_own_entry():
    text = experience_text(MISTAKES, SOLVED, {"question_id": 1, "db_id": "shop"})
    assert "lesson 1" not in text and "SELECT wrong_1" not in text and "SELECT right_1" not in text
    assert "SELECT wrong_2" in text
    text = experience_text(MISTAKES, SOLVED, {"question_id": 10, "db_id": "shop"})
    assert "SELECT solved_10" not in text


def test_no_pairs_section_for_a_db_without_mistakes():
    text = experience_text(MISTAKES, SOLVED, {"question_id": 99, "db_id": "other"})
    assert "lesson 3" in text and "on this database" not in text


def test_conventions_switch(fixture_db):
    agent = ExperiencedAgent(None, mistakes=MISTAKES, solved=SOLVED, use_conventions=False,
                             db_path_for=lambda _: fixture_db)
    text = agent.system_prompt({"question_id": 99, "db_id": "shop"})[2]["text"]
    assert "lesson 1" in text and "SELECT wrong_1" in text and "Solved questions" not in text


def test_system_prompt_is_the_single_agents_plus_one_cached_block(fixture_db):
    question = {"question_id": 99, "db_id": "shop"}
    base = SingleAgent(None, db_path_for=lambda _: fixture_db).system_prompt(question)
    blocks = ExperiencedAgent(None, mistakes=MISTAKES, solved=SOLVED,
                              db_path_for=lambda _: fixture_db).system_prompt(question)
    assert blocks[:2] == base
    assert blocks[2]["cache_control"] == {"type": "ephemeral"} and "lesson 1" in blocks[2]["text"]
