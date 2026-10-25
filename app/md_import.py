"""Markdown 批量导入：解析 + 落库。

格式见 parse_markdown docstring；导入编排逻辑在 import_parsed()。
"""
import re

from . import db


def _clean_question(t: str) -> str:
    return re.sub(r"^\d+[、.．)]\s*", "", t.strip())


def parse_markdown(text: str) -> dict:
    """解析导入格式，返回 {sections, errors, total}。

    # 一级标题 -> L1 大类（同名合并）
    > 引用行  -> L1 描述（紧跟标题的第一个引用）
    ## 二级标题 -> QA 问题（自动去 "12、" 序号）
    - 别名: / - 答案: / - 追问: / - STAR: / - 质量:  字段行，可重复
    其余非空文本 -> 追加到当前问题的答案正文
    """
    sections: list[dict] = []
    errors: list[str] = []
    cur_sec = None
    cur_q = None
    body_buf: list[str] = []

    def flush_body():
        if cur_q is not None and body_buf:
            extra = "\n\n".join(x.strip() for x in body_buf if x.strip())
            cur_q["answer"] = (cur_q["answer"] + "\n\n" + extra).strip() if cur_q["answer"] else extra
            body_buf.clear()

    field_pat = re.compile(r"^[-*]\s*(别名|答案|追问|STAR|[Ss]tar要点|质量)\s*[:：]\s*(.*)$")

    for ln in text.split("\n"):
        h1 = re.match(r"^#\s+(.+)$", ln)
        h2 = re.match(r"^##\s+(.+)$", ln)
        quote = re.match(r"^>\s?(.*)$", ln)
        if h1:
            flush_body()
            name = h1.group(1).strip()
            cur_sec = next((s for s in sections if s["name"] == name), None)
            if cur_sec is None:
                cur_sec = {"name": name, "description": "", "questions": []}
                sections.append(cur_sec)
            cur_q = None
        elif h2:
            flush_body()
            if cur_sec is None:
                errors.append(f"「{h2.group(1).strip()}」出现在任何 # 大类之前，已忽略")
                continue
            cur_q = {
                "question": _clean_question(h2.group(1)),
                "question_aliases": "", "answer": "", "follow_ups": "", "star_hint": "", "quality": 3,
            }
            cur_sec["questions"].append(cur_q)
        elif quote and cur_q is None and cur_sec is not None and not cur_sec["description"]:
            cur_sec["description"] = quote.group(1).strip()
        else:
            if cur_q is not None:
                m = field_pat.match(ln)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    if key == "别名":
                        cur_q["question_aliases"] = (cur_q["question_aliases"] + "\n" + val).strip("\n")
                    elif key == "答案":
                        cur_q["answer"] = (cur_q["answer"] + "\n" + val).strip()
                    elif key == "追问":
                        cur_q["follow_ups"] = (cur_q["follow_ups"] + "\n" + val).strip("\n")
                    elif key in ("STAR", "star要点"):
                        cur_q["star_hint"] = (cur_q["star_hint"] + "\n" + val).strip("\n")
                    elif key == "质量":
                        try:
                            v = int(val)
                            if 1 <= v <= 5:
                                cur_q["quality"] = v
                            else:
                                errors.append(f"「{cur_q['question'][:20]}」质量值非法（{val}，需1-5），已用默认3")
                        except ValueError:
                            errors.append(f"「{cur_q['question'][:20]}」质量值非法（{val}），已用默认3")
                    continue
            body_buf.append(ln)

    flush_body()
    total = sum(len(s["questions"]) for s in sections)
    return {"sections": sections, "errors": errors, "total": total}


def import_parsed(kb_id: int, parsed: dict) -> dict:
    """把 parse_markdown 的结果写入数据库：建缺失的 L1/L2，导入 QA（跳过同库同问题）。"""
    from .main import QAIn, create_qa  # 延迟导入避免循环依赖

    conn = db.get_conn()
    created_l1, created_l2, imported, skipped, failed = 0, 0, 0, 0, 0

    for sec in parsed["sections"]:
        row = conn.execute(
            "SELECT id FROM intent_l1 WHERE kb_id=? AND name=?", (kb_id, sec["name"])).fetchone()
        if row:
            l1_id = row["id"]
        else:
            with db.tx() as t:
                cur = t.execute(
                    "INSERT INTO intent_l1(kb_id, name, description) VALUES(?,?,?)",
                    (kb_id, sec["name"], sec["description"]))
                l1_id = cur.lastrowid
            created_l1 += 1

        # 导入专用 L2（不存在则建），避免把批量题塞进手工管理的子意图
        l2row = conn.execute(
            "SELECT id FROM intent_l2 WHERE l1_id=? AND name=?", (l1_id, "导入题集")).fetchone()
        if l2row:
            l2_id = l2row["id"]
        else:
            with db.tx() as t:
                cur = t.execute(
                    "INSERT INTO intent_l2(l1_id, name, description) VALUES(?,?,?)",
                    (l1_id, "导入题集", f"「{sec['name']}」批量导入的题目"))
                l2_id = cur.lastrowid
            created_l2 += 1

        existing = {
            r["question"] for r in conn.execute(
                "SELECT qa.question FROM qa JOIN intent_l2 l2 ON qa.l2_id=l2.id "
                "JOIN intent_l1 l1 ON l2.l1_id=l1.id WHERE l1.kb_id=?", (kb_id,)).fetchall()}

        for q in sec["questions"]:
            if not q["question"]:
                continue
            if q["question"] in existing:
                skipped += 1
                continue
            try:
                create_qa(QAIn(
                    l2_id=l2_id, question=q["question"], question_aliases=q["question_aliases"],
                    answer=q["answer"] or "（待补充）", follow_ups=q["follow_ups"],
                    star_hint=q["star_hint"], quality=q["quality"],
                ))
                imported += 1
                existing.add(q["question"])
            except Exception:
                failed += 1

    return {
        "imported": imported, "skipped_exist": skipped, "failed": failed,
        "created_l1": created_l1, "created_l2": created_l2,
        "errors": parsed["errors"], "total": parsed["total"],
    }
