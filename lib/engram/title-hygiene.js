/**
 * title-hygiene — 记忆标题卫生（零依赖纯函数；出货产物 lib/engram/title-hygiene.js 可被测试直读）
 *
 * 病案（2026-09-11 图谱实扫）：3021 节点里有 121 条标题以「：」或「*」收尾——
 * `交付报告：` `建议下一步**：` `session 层：` `按这个理解开做：**` ……
 * 全部出自 relay.ts 的 fallbackDistill：它把「最近一条消息」的末 200 字按标点切第一段当标题，
 * 而 `：` `*` 不在切分符里，于是 markdown 小标题/加粗引导句整条成了图谱入口。
 * 这种入口三个月后的自己看不出任何信息——正是「莫名其妙的记忆」的字面形态。
 *
 * 两道契约（engram/store.ts 实测口径）：
 *   ① `e.title = _t.slice(0, 12) || '对话片段'`——加载时**硬砍 12 字**，空标题退化成「对话片段」
 *      （图谱里真躺着 175 条「对话片段」，就是这条兜底造的）；
 *   ② 标题是入口锚点：供人扫、供 [[链接]] 指——尾部悬标点/太短/占位词都不成入口。
 *
 * 结论：清洗 + 封顶 12 字（带省略号时恰 12）；洗完不成入口的**宁可不记**（[[记忆宁缺毋滥]]）。
 */
/** engram store 加载时的标题硬上限（slice(0, 12)） */
export const TITLE_MAX = 12;
/** 入口标题的最小有效长度：短于此不值得当锚点 */
export const TITLE_MIN = 4;
/** 占位词（store 的兜底值与模型的自述词——都不是入口） */
const PLACEHOLDERS = new Set([
    '对话片段', '对话', '助手', '用户', 'assistant', 'user', 'ai', 'note', '摘要', '总结',
    '记录', '内容', '结果', '汇报', '报告', '说明', '如下', '以上',
]);
/**
 * 清洗一个蒸馏标题。返回空串 = **这条不该入图**（调用方应当 continue，而不是塞兜底词）。
 * 纯函数：无 IO、无时钟、无随机。
 */
export function sanitizeMemoryTitle(raw) {
    let t = String(raw ?? '').trim();
    if (!t)
        return '';
    // 行首 markdown/列表/表格符 + emoji（`## 部署` / `- 结论` / `| **三者全上**`——最后一条是本单
    // 清扫后仍在盘上的真样本：表格行整条被当成入口标题）
    t = t.replace(/^[\s#>*\-`~_|]+/u, '');
    t = t.replace(/^[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}]+\s*/u, '');
    // 强调符与表格竖线：入口里只会碍事（`**三者全上**` → `三者全上`）
    t = t.replace(/[*_`|]+/gu, '');
    // 尾部标点/空白（案底本体：`: ** ：` 都是从这里漏进去的）
    t = t.replace(/[：:，,、。.;；!！?？\s]+$/u, '');
    // 结构性括号字符（`[[标题]]`、`(备注)`）
    t = t.replace(/[[\]()（）#]/g, '');
    // 行首引导词（旧 fallback 的 `询问/请问/查找…`）
    t = t.replace(/^[询问请问查找查询关于]+/iu, '');
    t = t.replace(/^[\s#>*\-`~_|*]+/u, '').trim();
    if (!t)
        return '';
    if (PLACEHOLDERS.has(t.toLowerCase()))
        return '';
    // 封顶：超 12 字带省略号（恰 12），否则加载时被硬砍成半句
    if ([...t].length > TITLE_MAX)
        t = [...t].slice(0, TITLE_MAX - 1).join('') + '…';
    return [...t].length >= TITLE_MIN ? t : '';
}
/** 判「这个标题成不成入口」（给闸/观测面复用，与 sanitize 同口径）。 */
export function isUsableTitle(title) {
    const t = String(title ?? '').trim();
    if (!t)
        return false;
    if (/[*：:]\s*$/u.test(t))
        return false;
    if (PLACEHOLDERS.has(t.toLowerCase()))
        return false;
    return [...t].length >= TITLE_MIN;
}
/**
 * 标题 n-gram（蒸馏去重判定用）——**必须覆盖非中文标题**。
 *
 * 病案（2026-09-11 实测抓到，图谱里反复冒出的成对节点）：蒸馏的冗余去重原先把 gram
 * 只从 `[\u4e00-\u9fff]+` 里取，于是**纯拉丁标题**（`agent teams…` / `See-through…` /
 * `gpt-image-2…`）的 gram 集为空，去重扫描被 `if (_tgrams.size > 0)` 整体跳过
 * → 只要有非中文标题的记忆，**每次蒸馏都新增一条**，同一话题反复成对堆积。
 * 修法：CJK 走二元组（原行为不变），非 CJK 补**词级 token**（小写、长度≥2）。
 */
export function titleGrams(title) {
    const s = String(title ?? '');
    const out = new Set();
    for (const seg of (s.match(/[\u4e00-\u9fff]+/g) ?? [])) {
        for (let i = 0; i < seg.length - 1; i++)
            out.add(seg.slice(i, i + 2));
    }
    for (const w of (s.toLowerCase().match(/[a-z0-9][a-z0-9_-]+/g) ?? []))
        out.add(w);
    return out;
}
