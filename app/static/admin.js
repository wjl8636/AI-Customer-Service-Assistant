/*
  来源：公众号@小林coding
  后端八股网站：xiaolincoding.com
  Agent网站：xiaolinnote.com
  简历模版：jianli.xiaolinnote.com
*/
/* 后台管理外壳:一份导航挂在所有后台页上(知识库录入 / RAG 评估 / 待审队列 / 观测与成本 /
   主题分布 / 分类器验收)。

   这些页面分属不同章、样式各自内联,所以导航自带样式、自己注入,不去动宿主页的 CSS;
   整个文件包在 IIFE 里,只往 window 上挂两个名字,免得和宿主页的 $ / el 撞名。
   模块入口都是各章原本的路径,导航只是把它们收到一处,不做跳转改写——文档里贴的链接照样能用。 */
(function () {
  const NAV = [
    { href: "/admin", label: "后台首页" },
    { href: "/kb", label: "知识库录入" },
    { href: "/rag-eval", label: "RAG 评估" },
    { href: "/review", label: "飞轮待审" },
    { href: "/observability", label: "观测与成本" },
    { href: "/topics", label: "主题分布" },
    {
      href: "/acceptance", label: "分类器验收",
      children: [
        ["/acceptance", "总览"],
        ["/acceptance/eval", "评测详情"],
        ["/acceptance/data", "数据产物"],
        ["/acceptance/errors", "错例复核"],
      ],
    },
  ];

  const CSS = `
  .mh-nav { margin-top: 12px; }
  .mh-nav .row {
    display: flex; flex-wrap: wrap; align-items: stretch;
    border: 3px solid #2b2632; background: #fffaf0; box-shadow: 3px 3px 0 #2b2632;
  }
  .mh-nav .row a {
    display: flex; align-items: center; gap: 6px; padding: 7px 13px; font-size: 13px;
    text-decoration: none; color: #2b2632; border-right: 3px solid #2b2632; font-weight: 700;
  }
  .mh-nav .row a:last-child { border-right: 0; }
  .mh-nav .row a.on { background: #ff9f57; }
  .mh-nav .row .brand {
    display: flex; align-items: center; padding: 7px 12px; font-size: 12px;
    background: #2b2632; color: #fff6e6; border-right: 3px solid #2b2632; letter-spacing: 1px;
  }
  .mh-nav .row .grow { flex: 1; border-right: 3px solid #2b2632; }
  .mh-nav .sub {
    display: flex; flex-wrap: wrap; border: 3px solid #2b2632; border-top: 0;
    background: #fff6e6; box-shadow: 3px 3px 0 #2b2632;
  }
  .mh-nav .sub a {
    padding: 5px 12px; font-size: 12px; text-decoration: none; color: #2b2632;
    border-right: 3px solid #2b2632;
  }
  .mh-nav .sub a:last-child { border-right: 0; }
  .mh-nav .sub a.on { background: #ff9f57; font-weight: 700; }
  @media (max-width: 640px) { .mh-nav .row .grow { display: none; } }
  `;

  function injectCss() {
    if (document.getElementById("mh-nav-css")) return;
    const s = document.createElement("style");
    s.id = "mh-nav-css";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  function link(href, label, on, cls) {
    const a = document.createElement("a");
    a.href = href;
    a.className = (on ? "on " : "") + (cls || "");
    a.textContent = label;
    return a;
  }

  /** 当前页归属哪个模块:精确命中优先,其次按前缀(/acceptance/eval 归 /acceptance)。 */
  function moduleOf(active) {
    return NAV.find((m) => m.href === active)
      || NAV.find((m) => m.href !== "/" && active.startsWith(m.href + "/"));
  }

  function renderAdminNav(active) {
    injectCss();
    const wrap = document.createElement("div");
    wrap.className = "mh-nav";
    const row = document.createElement("div");
    row.className = "row";
    const brand = document.createElement("span");
    brand.className = "brand";
    brand.textContent = "后台管理";
    row.appendChild(brand);
    const mod = moduleOf(active);
    for (const m of NAV) row.appendChild(link(m.href, m.label, m === mod));
    const grow = document.createElement("span");
    grow.className = "grow";
    row.appendChild(grow);
    row.appendChild(link("/", "聊天页 →", false));
    wrap.appendChild(row);

    if (mod && mod.children) {
      const sub = document.createElement("div");
      sub.className = "sub";
      for (const [href, label] of mod.children) sub.appendChild(link(href, label, href === active));
      wrap.appendChild(sub);
    }
    return wrap;
  }

  /** 挂到顶栏底下(顶栏是每页自己的标题条);找不到顶栏就摆在 .wrap 最前面。 */
  function mountAdminNav(active) {
    const nav = renderAdminNav(active);
    const bar = document.querySelector(".topbar");
    if (bar && bar.parentNode) bar.parentNode.insertBefore(nav, bar.nextSibling);
    else {
      const w = document.querySelector(".wrap") || document.body;
      w.insertBefore(nav, w.firstChild);
    }
    return nav;
  }

  window.renderAdminNav = renderAdminNav;
  window.mountAdminNav = mountAdminNav;
})();
