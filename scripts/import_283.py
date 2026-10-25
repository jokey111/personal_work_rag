"""把 283 题 payload 导入远端 personal_work_rag 服务。

流程：登录 → （可选）建知识库 → 每大类建 L1 → 每类下建一个"L2 综合题集" → 批量导入 QA。
用法：python import_283.py http://43.161.218.189:8090 admin <密码> payload_283.json [--kb-name 面试283题]
"""
import json
import sys
import urllib.request

BASE = sys.argv[1]
USER, PASS = sys.argv[2], sys.argv[3]
PAYLOAD = sys.argv[4]
KB_NAME = "面试283题库"

if "--kb-name" in sys.argv:
    KB_NAME = sys.argv[sys.argv.index("--kb-name") + 1]


def call(method: str, path: str, token: str = "", body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


# 1. 登录
tok = call("POST", "/api/login", body={"username": USER, "password": PASS})["token"]
print("登录 OK")

# 2. 知识库：找"面试283题库"，没有则建
kbs = call("GET", "/api/kbs", tok)
kb = next((k for k in kbs if k["name"] == KB_NAME), None)
if not kb:
    kb_id = call("POST", "/api/kbs", tok, {"name": KB_NAME, "description": "AI产品经理面试283题（飞书06期），答案为参考思路，待个人化"})["id"]
    print(f"新建知识库 {KB_NAME} id={kb_id}")
else:
    kb_id = kb["id"]
    print(f"复用知识库 {KB_NAME} id={kb_id}")

with open(PAYLOAD, encoding="utf-8") as f:
    payload = json.load(f)

# 3. 每大类：建 L1（如已存在则复用）+ L2"综合题集"
tree = call("GET", f"/api/intents?kb_id={kb_id}", tok)
existing_l1 = {t["name"]: t["id"] for t in tree}
l2_by_l1 = {}
for t in tree:
    for c in t.get("children", []):
        l2_by_l1[t["id"]] = c["id"]  # 每类先只取第一个 L2

l1_map: dict[str, int] = {}
for sec in payload["sections"]:
    name = sec["l1_name"]
    if name in existing_l1:
        l1_id = existing_l1[name]
    else:
        l1_id = call("POST", "/api/intents/l1", tok, {
            "kb_id": kb_id, "name": name,
            "description": f"{name}（AI产品经理面试题库）",
        })["id"]
        print(f"  L1 已建: {name}")
    l1_map[name] = l1_id
    if l1_id not in l2_by_l1:
        l2_by_l1[l1_id] = call("POST", "/api/intents/l2", tok, {
            "l1_id": l1_id, "name": "综合题集",
            "description": f"{name}下的面试题汇总",
        })["id"]

# 4. 查已有 QA（幂等：跳过已存在的同问题）
existing_q = set()
for sec in payload["sections"]:
    rows = call("GET", f"/api/qas?kb_id={kb_id}&q=", tok)  # 拉一次全量
    break
for r in rows:
    existing_q.add(r["question"])
print(f"库中已有 QA {len(existing_q)} 条")

# 5. 批量导入（逐大类调用 import 接口）
ok_total, skip_total = 0, 0
for sec in payload["sections"]:
    l2_id = l2_by_l1[l1_map[sec["l1_name"]]]
    items = [
        {"l2_id": l2_id, "question": q["question"], "answer": q["answer"] or "（待补充）", "status": "active"}
        for q in sec["questions"] if q["question"] not in existing_q
    ]
    if not items:
        print(f"  {sec['l1_name']}: 全部已存在，跳过")
        continue
    r = call("POST", "/api/qas/import", tok, {"items": items})
    ok_total += r["imported"]
    skip_total += r["total"] - r["imported"]
    print(f"  {sec['l1_name']}: 导入 {r['imported']}/{r['total']}")

print(f"\n完成：成功 {ok_total} 条，失败 {skip_total} 条")
stats = call("GET", f"/api/stats/overview?kb_id={kb_id}", tok)
print(f"题库现有 QA 总数: {stats['qa_total']}")
