import json, os
src = 'F:/dsh/.zcode/real-session2/session.jsonl'
out = 'F:/dsh/.zcode/recent-100k.md'
# 提取消息文本（user + assistant 的 text 块）
msgs = []
for line in open(src, encoding='utf-8'):
    if '"type":"user/message"' in line or '"type":"assistant/message"' in line:
        try:
            node = json.loads(line)
        except Exception:
            continue
        t = node.get('type', '')
        data = node.get('data', {})
        if t == 'user/message':
            c = data.get('content', [])
        else:
            c = data.get('message', {}).get('content', [])
        texts = []
        for b in c:
            if isinstance(b, dict):
                if b.get('type') == 'text' and b.get('text'):
                    texts.append(b['text'])
        if texts:
            role = '用户' if t == 'user/message' else '助手'
            msgs.append((node.get('time', 0), role, '\n'.join(texts)))
# 按时间排序，取最近的（目标 ~140k 字符 ≈ 100k token）
msgs.sort(key=lambda m: m[0])
chars = 0
recent = []
for t, role, text in reversed(msgs):
    block = f'\n### {role}\n{text}\n'
    recent.append((t, role, text))
    chars += len(block)
    if chars > 140000:
        break
recent.reverse()
with open(out, 'w', encoding='utf-8') as f:
    for t, role, text in recent:
        f.write(f'\n### {role}\n{text}\n')
print(f'消息数: {len(recent)}，字符: {chars}，文件: {out}')
