"""Markdown 批量导入：解析 + 落库。

格式见 parse_markdown docstring；导入编排逻辑在 import_parsed()。
"""
import re

from . import db


def _clean_question(t: str) -> str:
    return re.sub(r"^\d+[、.．)]\s*", "", t.strip())


def parse_markdown(text: str) -> dict:
    """解析导入格式，返回 {sections, errors, total}。

    # 一级标题 -> 一级分类（同名合并）
    > 引用行  -> 分类描述（仅占位，不落库）
    ## 二级标题 -> QA 问题（自动去 "12、" 序号）
    - 标签:   -> 该问题的标签（逗号/顿号分隔，可多行）
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

    field_pat = re.compile(r"^[-*]\s*(标签|别名|答案|追问|STAR|[Ss]tar要点|质量)\s*[:：]\s*(.*)$")

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
                "question_aliases": "", "answer": "", "follow_ups": "", "star_hint": "",
                "quality": 3, "tags": [],
            }
            cur_sec["questions"].append(cur_q)
        elif quote and cur_q is None and cur_sec is not None and not cur_sec["description"]:
            cur_sec["description"] = quote.group(1).strip()
        else:
            if cur_q is not None:
                m = field_pat.match(ln)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    if key == "标签":
                        parts = [p.strip() for p in re.split(r"[,，、;；]", val) if p.strip()]
                        cur_q["tags"].extend(parts)
                    elif key == "别名":
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
    """把 parse_markdown 的结果写入指定知识库。

    `# 大类` 标题 → 一级分类（不存在则创建）；`- 标签:` → 标签（不存在则创建）；
    QA 直接入库并挂分类与标签（跳过同库同问题）。
    """
    from .main import QAIn, create_qa, create_category, create_tag, CategoryIn, TagIn  # 延迟导入避免循环依赖

    conn = db.get_conn()
    imported, skipped, failed = 0, 0, 0
    created_categories, created_tags = 0, 0

    existing = {
        r["question"] for r in conn.execute(
            "SELECT question FROM qa WHERE kb_id=?", (kb_id,)).fetchall()}

    for sec in parsed["sections"]:
        # 大类 → 一级分类（不存在则建）
        row = conn.execute(
            "SELECT id FROM category WHERE kb_id=? AND parent_id IS NULL AND name=?",
            (kb_id, sec["name"])).fetchone()
        if row:
            cat_id = row["id"]
        else:
            res = create_category(CategoryIn(kb_id=kb_id, parent_id=None, name=sec["name"]))
            cat_id = res["id"]
            created_categories += 1

        # 本节涉及的标签一次性建好
        tag_ids_by_name: dict[str, int] = {}
        for q in sec["questions"]:
            for name in q.get("tags", []):
                if name in tag_ids_by_name:
                    continue
                trow = conn.execute("SELECT id FROM tag WHERE name=?", (name,)).fetchone()
                if trow:
                    tag_ids_by_name[name] = trow["id"]
                else:
                    res = create_tag(TagIn(name=name))
                    tag_ids_by_name[name] = res["id"]
                    created_tags += 1

        for q in sec["questions"]:
            if not q["question"]:
                continue
            if q["question"] in existing:
                skipped += 1
                continue
            try:
                create_qa(QAIn(
                    kb_id=kb_id, category_id=cat_id, question=q["question"],
                    question_aliases=q["question_aliases"],
                    answer=q["answer"] or "（待补充）", follow_ups=q["follow_ups"],
                    star_hint=q["star_hint"], quality=q["quality"],
                    tag_ids=[tag_ids_by_name[n] for n in q.get("tags", []) if n in tag_ids_by_name],
                ))
                imported += 1
                existing.add(q["question"])
            except Exception:
                failed += 1

    return {
        "imported": imported, "skipped_exist": skipped, "failed": failed,
        "created_categories": created_categories, "created_tags": created_tags,
        "errors": parsed["errors"], "total": parsed["total"],
    }
