/*
  来源：公众号@小林coding
  后端八股网站：xiaolincoding.com
  Agent网站：xiaolinnote.com
  简历模版：jianli.xiaolinnote.com
*/
/* ch10 验收页共享脚本:四个页面公用的取数、提示、导航、以及「重跑」按钮那一套。

重跑的交互契约:POST 发起 → 每 1.2s 轮询状态与日志尾 → 收到终态(ok/failed/stopped)
停止轮询、回调页面重新取数。作业名是后端白名单里的常量,前端只传名字。 */

const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function toast(msg, isErr) {
  let t = $("toast");
  if (!t) { t = el("div"); t.id = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.className = isErr ? "err" : "";
  t.style.display = "block";
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.style.display = "none"; }, 3600);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || ("HTTP " + r.status));
  return body;
}

/* 页间导航在 admin.js 的 mountAdminNav:后台各页共用一份,知识库录入与飞轮那几页也在里头。 */

const fmtTime = (iso) => {
  if (!iso) return "—";
  return String(iso).replace("T", " ").slice(0, 19);
};

const fmtBytes = (n) => {
  if (!n && n !== 0) return "—";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MB";
  return (n / 1024 / 1024 / 1024).toFixed(2) + " GB";
};

/** 三位小数 + 一条横条:F1 这类 0~1 的分数统一这样画,能扫出高低。
    hi/lo 只在给了红线时才上色——没有线就不该有"及格/不及格"的暗示。 */
function scoreCell(v, redLine) {
  const td = el("td", "num");
  const box = el("div", "cell-bar");
  const val = el("span", "v", v === null || v === undefined ? "—" : v.toFixed(3));
  const track = el("span", "track");
  const fill = el("span", "fill");
  fill.style.width = Math.max(2, Math.round((v || 0) * 100)) + "%";
  if (redLine !== null && redLine !== undefined) fill.classList.add(v >= redLine ? "hi" : "lo");
  track.appendChild(fill);
  box.appendChild(val);
  box.appendChild(track);
  td.appendChild(box);
  return td;
}

const STATUS_LABEL = {
  idle: "未跑过", running: "运行中", ok: "已完成", failed: "失败", stopped: "已停止",
};

/* 日志尾缓存:作业跑完会回调页面重新取数,取数把按钮和日志窗口整个重建。
   缓存按作业名留着,重建时贴回去——不然日志恰好在跑完那一刻消失,结论就读不到了。 */
const JOB_LOGS = {};

/**
 * 一个「重跑」按钮 + 共用日志窗口。
 * @param spec 后端 /api/jobs 里的一项(含 name/title/cmd/needs/heavy/status)
 * @param logbox 日志显示的 <pre>
 * @param onFinish 作业进终态后的回调(通常是页面重新取数)
 */
function jobButton(spec, logbox, onFinish) {
  const btn = el("button", "btn sm go", "重跑 " + spec.title);
  btn.title = spec.cmd + (spec.needs && spec.needs !== "—" ? "(" + spec.needs + ")" : "");
  let timer = null;

  function paint(st) {
    const running = st.status === "running";
    btn.disabled = running;
    btn.textContent = running
      ? "运行中… " + spec.title
      : (st.status === "idle" ? "重跑 " : "再跑一次 ") + spec.title;
    if (st.log) {
      JOB_LOGS[spec.name] = st.log;
      showLog(st.log);
    }
  }

  function showLog(text) {
    logbox.classList.add("on");
    logbox.textContent = text;
    logbox.scrollTop = logbox.scrollHeight;
  }

  async function poll() {
    try {
      const st = await api("/api/jobs/" + spec.name);
      paint(st);
      if (st.status !== "running") {
        clearInterval(timer); timer = null;
        toast(spec.title + "：" + STATUS_LABEL[st.status]
          + (st.returncode !== null ? "(退出码 " + st.returncode + ")" : ""),
          st.status !== "ok");
        if (onFinish) onFinish();
      }
    } catch (e) {
      clearInterval(timer); timer = null;
      toast("轮询失败：" + e.message, true);
    }
  }

  btn.addEventListener("click", async () => {
    if (spec.heavy && !confirm(
      "「" + spec.title + "」是分钟级重活（" + spec.cmd + "）。\n"
      + (spec.needs && spec.needs !== "—" ? "前置：" + spec.needs + "\n" : "")
      + "确认现在跑？")) return;
    btn.disabled = true;
    logbox.classList.add("on");
    logbox.textContent = "发起中…";
    try {
      const st = await api("/api/jobs/" + spec.name, { method: "POST" });
      paint(st);
      timer = setInterval(poll, 1200);
    } catch (e) {
      btn.disabled = false;
      logbox.textContent = "发起失败：" + e.message;
      toast("发起失败：" + e.message, true);
    }
  });

  paint(spec);
  if (JOB_LOGS[spec.name]) showLog(JOB_LOGS[spec.name]);   // 重建后把日志贴回去
  // 页面打开时作业正在跑(上一个标签页发起的):直接接上轮询,别让它看着像卡住
  if (spec.status === "running") timer = setInterval(poll, 1200);
  return btn;
}

/** 一排作业按钮 + 它们共用的一个日志窗口。 */
function jobRow(specs, onFinish, extraNote) {
  const wrap = el("div");
  const row = el("div", "jobrow");
  const logbox = el("pre", "logbox");
  for (const s of specs) row.appendChild(jobButton(s, logbox, onFinish));
  if (extraNote) row.appendChild(el("span", "needs", extraNote));
  wrap.appendChild(row);
  wrap.appendChild(logbox);
  return wrap;
}

/** 读图小注:产物里带着模型看这一轮数写的那句就用它,数字自动加粗;没有才用页面写死的兜底句。
 *  注是脚本落盘时生成并校过数的(app/core/read_notes.py),页面这边只负责显示,不再自己下结论。 */
function readNoteHtml(note, fallbackHtml) {
  if (!note) return fallbackHtml;
  const esc = String(note).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  // 只加粗真正的数,名字里的数字不算(BM25 的 25、Recall@10 的 10、bge-m3 的 3),
  // 与 read_notes.py 里那条校验用的边界规则保持一致
  return esc.replace(/(?<![A-Za-z@_.\-\d])\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z_])/g,
    (n) => "<b>" + n + "</b>");
}

/** 产物缺失时的统一占位:说清楚缺什么、该跑哪个目标。 */
function missingBox(hint) {
  return el("div", "miss", hint || "产物还没生成，先跑对应的 make 目标");
}
